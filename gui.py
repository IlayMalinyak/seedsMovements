# __author:IlayK
# data:03/05/2022

import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utils import *
from registration import register, warp_image
import SimpleITK as sitk
from abc import ABC, abstractmethod
from analyze import assignment, calc_dist, analyze_distances, apply_transformation_on_seeds
import pandas as pd

px = 1/plt.rcParams['figure.dpi']
plt.rcParams["figure.figsize"] = (500*px,500*px)
font_header = ("Arial, 18")
font_text = ("Ariel, 12")
sg.theme("SandyBeach")

OPTIMIZERS = ['GD', 'LBFGS2']

NO_REGISTRATION_ERR = "No registration has been saved on this case yet. please choose registrarion" \
                      " type and run registrarion"


class BasicViewer(ABC):

    @abstractmethod
    def show(self):
        pass

    @abstractmethod
    def update_slice(self, val):
        pass


class DicomViewer(BasicViewer):
    """
    Dicom viewer class
    """
    def __init__(self, array, canvas, name):
        self.name = name
        self.array = array
        self.canvas = canvas
        # self.canvas.bind_all("<MouseWheel>", self.update_slice)
        self.fig, self.ax = plt.subplots(1, 1)
        self.figure_canvas_agg = FigureCanvasTkAgg(self.fig, self.canvas)
        # self.figure_canvas_agg.get_tk_widget().forget()
        if len(array.shape) == 3:
            rows, cols, self.slices = array.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = self.array.shape
        self.ind = self.slices//2
        axSlice = plt.axes([0.1, 0.18, 0.05, 0.63])
        self.slice_slider = Slider(
            ax=axSlice,
            label='Slice',
            valmin=0,
            valmax=self.slices - 1,
            valinit=self.ind,
            valstep=1, color='magenta',
            orientation="vertical"
        )
        self.slice_slider.on_changed(self.update_slice)
        plt.subplots_adjust(left=0.25, bottom=0.1)

    def show(self):
        self.im = self.ax.imshow(self.array[:,:,self.ind], cmap='gray')
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both')

    def update_slice(self, val):
        # print("scroll event on ", self.name)
        self.ind = int(self.slice_slider.val)
        # print(self.ind, event.y)
        self.im.set_data(self.array[..., self.ind])
        self.im.axes.figure.canvas.draw()
        # self.show()

    def set_array(self, array):
        self.array = array

    def clear(self):
        self.figure_canvas_agg.get_tk_widget().forget()


class OverlayViewer(DicomViewer):
    """
    dicoms overlay viewer class
    """
    def __init__(self, array1, array2, canvas, name):
        super(OverlayViewer, self).__init__(array1, canvas, name)
        self.array2 = array2
        self.alpha = 0.3
        axAlpha = plt.axes([0.25, 0.02, 0.63, 0.05])
        self.alpha_slider = Slider(
            ax=axAlpha,
            label='Alpha',
            valmin=0,
            valmax=1,
            valinit=self.alpha,
            color='magenta',
        )
        self.alpha_slider.on_changed(self.update_alpha)
        plt.subplots_adjust(left=0.25, bottom=0.1)

    def show(self):
        self.im = self.ax.imshow(self.array[:, :, self.ind], cmap='gray')
        if self.ind < self.array2.shape[-1]:
            self.overlay = self.ax.imshow(self.array2[..., self.ind], cmap='Reds', interpolation='none',
                                 alpha=self.alpha)
        else:
            self.overlay = self.ax.imshow(self.array2[..., -1], cmap='Reds', interpolation='none',
                                          alpha=self.alpha)
        self.figure_canvas_agg.draw()
        self.figure_canvas_agg.get_tk_widget().pack(side='top', fill='both')

    def update_slice(self, val):
        self.ind = int(self.slice_slider.val)
        # print(self.ind, event.y)
        self.im.set_data(self.array[..., self.ind])
        self.overlay.set_data(self.array[..., self.ind])
        self.im.axes.figure.canvas.draw()
        # self.show()

    def update_alpha(self, val):
        self.alpha = self.alpha_slider.val
        self.overlay.set_alpha(self.alpha)

    def set_arrays(self, arr1, arr2):
        self.array = arr1
        self.array2 = arr2


class App():
    """
    GUI class
    """
    def __init__(self):
        data_column = [
            [sg.Text('Experiment name ', size=(15, 1), font=font_header)],
            [sg.InputText(key='-NAME-', enable_events=True)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Upload files', font=font_header)],
            [sg.Text('DICOM dir 1', font=font_text), sg.In(size=(15, 1), enable_events=True, key='-FIXED_FOLDER-'),
             sg.FolderBrowse(font=font_text)],
            [sg.Text('DICOM dir 2', font=font_text), sg.In(size=(15, 1), enable_events=True, key='-MOVING_FOLDER-'),
             sg.FolderBrowse(font=font_text)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text('Registration', font=font_header)],
            [sg.Text("Registration Type", font=font_text), sg.Combo(['Use saved Registration', 'Rigid', 'Bspline'], enable_events=True, font=font_text,
                                                                    key="-REG_MENU-")],
            [sg.Text('Select Optimizer', key='-OPT_TEXT-', visible=False),
             sg.Combo(['LBFGS2', 'Gradient Decent'], key='-OPT_MENU-', visible=False, enable_events=True),
             sg.Text('Select Metric', key='-METRIC_TEXT-', visible=False),
             sg.Combo(['Mean Squares', 'Mutual information'], key='-METRIC_MENU-', visible=False, enable_events=True)],
             [sg.Text('Sampling percentage (0-1)', key='-GLOBAL_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GLOBAL_PARAM_1-', enable_events=True),
             sg.Text('Number of Iterations', key='-GLOBAL_PARAM_TEXT_2-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GLOBAL_PARAM_2-', enable_events=True)
             ],
            [
             sg.Text('Convergence Tolerance', key='-GLOBAL_PARAM_TEXT_3-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GLOBAL_PARAM_3-', enable_events=True)
             ,sg.Text('Solution accuracy', key='-LBFGS2_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-LBFGS2_PARAM_1-', enable_events=True),
             sg.Text('Learning Rate', key='-GD_PARAM_TEXT_1-', visible=False),
             sg.InputText(size=(5, 1), visible=False, key='-GD_PARAM_1-', enable_events=True),
             ],
            [sg.Button("Run Registration", font=font_text, key='-REGISTER-')],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Button("Show Registration Metric", font=font_text, key='-PLOT_REG-')],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Button("Show overlay", font=font_text, key='-OVERLAY-')],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Assign Pairs", font=font_header)],
            [sg.Combo(['Munkres', 'Upload List'], enable_events=True, font=font_text,
                                                                    key="-ASSIGN_MENU-")],
            [sg.Text('Upload assignment file', font=font_text, key='-ASSIGN_TEXT-', visible=False),
             sg.In(size=(15, 1), enable_events=True, key='-ASSIGN_INPUT-', visible=False),
             sg.FileBrowse(font=font_text, visible=False, key='-ASSIGN_BROWSER-')],
            [sg.Button("Assign", font=font_text, key="-ASSIGN-")],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Calculate movements", font=font_header)],
            [sg.Button("Calculate", font=font_text, key='-CALC_MOVE-')],
            [sg.Button("Show movements", font=font_text, key='-SHOW_MOVE-', visible=False), sg.Button("Show Pairs",
                                                                        font=font_text, key='-SHOW_PAIRS-', visible=False)
             ,sg.Button("Save results to csv", font=font_text, key='-SAVE-', visible=False)],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Text("Massages Window", font=font_header)],
            [sg.Text("", font=font_header, size=(40, 7), key="-MSG-", background_color="white",
                     text_color='red')],
            [sg.HSeparator()],
            [sg.VPush()],
            [sg.VPush()],
            [sg.Exit(font=font_text)]]

        img_column = [[sg.Text('Dicom 1', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-FIXED_CANVAS-')],
                      [sg.Text('Dicom 2', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-MOVING_CANVAS-')]
                       ]
        results_column = [[sg.Text('Overlay', font=font_header)],
                      [sg.Canvas(size=(400, 400), key='-OVERLAY_CANVAS-')],
                      [sg.Text('Results', font=font_header)],
                      [sg.Image("", key='-IMAGE-')]
                       ]

        layout = [
            [sg.Column(data_column, vertical_alignment='top', element_justification='c'),
             sg.VSeperator(),
             sg.Column(img_column, element_justification='c'),
            sg.VSeperator(),
            sg.Column(results_column, vertical_alignment='top', element_justification='c')]]

        self.window = sg.Window('Demo Application - registration', layout, finalize=True,
                            return_keyboard_events=True)
        self.case_name = None
        self.fixed_dict = None
        self.moving_dict = None
        self.fixed_array = None
        self.moving_array = None
        self.fixed_viewer = None
        self.moving_viewer = None
        self.overlay_viewer = None

        self.registration_type = None
        self.registration_params = {}
        self.optimizer = None
        self.local_params = {}
        self.warped_moving = None
        self.tfm = None

        self.assignment_method = None
        self.fixed_seeds, self.fixed_orientation = None, None
        self.moving_seeds, self.moving_orientation = None, None
        self.seeds_tips_fixed = None
        self.seeds_tips_moving = None
        self.seeds_tips_moving_warped = None
        self.fixed_idx, self.warped_moving_idx = None, None
        self.assignment_dists = [None]
        self.errors = [None]

        self.df_features = ["Timestamp", "Experiment", "Registration Method", "Optimizer", "Metric", "Number Of Iterations", "Learning Rate",
                            "Accuracy Threshold", "Convergence Delta" ,"Average Movement (mm)", "Median Movement (mm)",
                            "Standard Deviation (mm)", "Maximum Movement (mm)", "Average Error (mm)"]
        self.create_empty_param_dict()

    def update_massage(self, massage):
        self.window['-MSG-'].update(massage)

    @staticmethod
    def find_files_with_extention(folder, extention):
        files = os.listdir('./registration_output/{}'.format(folder))
        return [i for i in files if i.endswith(extention)]

    def run(self):
        # app = None
        # event, values = self.window.read()
        # print(values)
        # DicomCanvas(fixed, window['-CANVAS-'].TKCanvas)
        while True:
            event, values = self.window.read()
            # print("size - ", self.window.size)
            # print(event)
            if event == "-NAME-":
                self.case_name = values[event]
            if event == '-FIXED_FOLDER-':
                # print('fixed folder', values['-FIXED_FOLDER-'])
                self.fixed_dict = get_all_dicom_files(values['-FIXED_FOLDER-'], {})
                self.fixed_array = read_dicom(self.fixed_dict['CT'])
                if self.fixed_viewer is None:
                    self.fixed_viewer = DicomViewer(self.fixed_array, self.window['-FIXED_CANVAS-'].TKCanvas, 'fixed')
                else:
                    self.fixed_viewer.clear()
                    self.fixed_viewer.set_array(self.fixed_array)
                self.fixed_viewer.show()
                self.update_massage("{} was uploaded successfully".format(self.fixed_dict['meta']['ID'].value))
            if event == '-MOVING_FOLDER-':
                # print('fixed folder', values['-FIXED_FOLDER-'])
                self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
                self.moving_array = read_dicom(self.moving_dict['CT'])
                if self.moving_viewer is None:
                    self.moving_viewer = DicomViewer(self.moving_array, self.window['-MOVING_CANVAS-'].TKCanvas, 'moving')
                else:
                    self.moving_viewer.clear()
                    self.moving_viewer.set_array(self.moving_array)
                self.moving_viewer.show()
                self.update_massage("{} was uploaded successfully".format(self.moving_dict['meta']['ID'].value))
            if event == "-REG_MENU-":
                self.registration_type = values[event] if values[event] != 'Use saved Registration' else None
            self.show_relevant_params_input()
            self.create_param_dicts(values)
            if event == '-REGISTER-':
                if self.case_name not in os.listdir('./registration_output'):
                    dir_name = f"./registration_output/{self.case_name}"
                    os.mkdir(dir_name)
                if self.case_name not in os.listdir('./movement_output'):
                    dir_name = f"./movement_output/{self.case_name}"
                    os.mkdir(dir_name)
                if self.registration_type is not None:
                    if self.fixed_array is not None and self.moving_array is not None:
                        if self.case_name is not None:
                            self.update_massage("Registering...\nThis may take a while")
                            self.window.refresh()
                            self.fixed_array_itk, self.moving_array_itk, self.warped_moving, self.tfm = register(self.moving_dict['CT'],
                                        self.fixed_dict['CT'], './registration_output/{}/{}.tfm'.format(self.case_name, self.registration_type),
                                       self.registration_type, self.registration_params)
                            self.update_massage("Registration finished. outputs was saved at:\n {}".format(
                                './registration_output/{}'.format(self.case_name)))
                        else:
                            self.update_massage("Enter Experiment Name")
                    else:
                        self.update_massage("You must upload 2 DICOMS to perform registration")
                else:
                    self.use_saved_transformation()
            if event == "-PLOT_REG-":
                self.plot_name = self.plot_registration()
                if self.plot_name is None:
                    self.update_massage("Nothing to show")
            if event == "-OVERLAY-":
                if self.warped_moving is not None:
                    if self.overlay_viewer is None:
                        self.overlay_viewer = OverlayViewer(self.fixed_array, self.warped_moving, self.window['-OVERLAY_CANVAS-'].TKCanvas,
                                                        'overlay')
                    else:
                        self.overlay_viewer.clear()
                        self.overlay_viewer.set_arrays(self.fixed_array, self.warped_moving)
                    self.overlay_viewer.show()
            if event == "-ASSIGN_MENU-":
                self.assignment_method = values[event]
                self.fixed_seeds, self.fixed_orientation = get_seeds_dcm(self.fixed_dict["RTPLAN"])
                self.moving_seeds, self.moving_orientation = get_seeds_dcm(self.moving_dict["RTPLAN"])
                meta_fixed = self.fixed_dict['meta']
                meta_moving = self.moving_dict['meta']
                self.seeds_tips_fixed = get_seeds_tips(self.fixed_seeds, self.fixed_orientation,
                                        meta_fixed['pixelSpacing'][0], meta_fixed['pixelSpacing'][1],
                                        meta_fixed['sliceThickness'])
                self.seeds_tips_moving = get_seeds_tips(self.moving_seeds, self.moving_orientation,
                                        meta_moving['pixelSpacing'][0], meta_moving['pixelSpacing'][1],
                                        meta_moving['sliceThickness'])

            if event == '-ASSIGN-':
                if self.assignment_method is None:
                    self.update_massage("Please choose assignment method first")
                else:
                    meta_fixed = self.fixed_dict['meta']
                    meta_moving = self.moving_dict['meta']
                    self.seeds_tips_moving_warped = apply_transformation_on_seeds(self.tfm, self.seeds_tips_moving)
                    if self.assignment_method == "Munkres":
                        self.fixed_idx, self.warped_moving_idx = assignment(self.seeds_tips_fixed, self.seeds_tips_moving_warped,
                                                           meta_fixed, meta_moving)
                    else:
                        path = values["-ASSIGN_INPUT-"]
                        self.fixed_idx, self.warped_moving_idx = self.parse_lists_from_file('./list')
                    self.update_massage(f'assignment is:\n{self.fixed_idx}\n{self.warped_moving_idx}')
                    # self.window['-MSG-'].update(font=font_text)

                # seeds1, seeds2, dists, errors = calculate_distances(case, seeds1, seeds2, meta_fixed, meta_moving, case_idx,
                #                                                     assignment)

            if self.assignment_method == "Upload List":
                self.show_assignment_uploader()
            if event == "-CALC_MOVE-":
                if self.fixed_idx is not None:
                    if len(self.fixed_idx) > self.seeds_tips_fixed.shape[-1] or len(self.warped_moving_idx) > \
                            self.seeds_tips_moving_warped.shape[-1]:
                        self.update_massage(f'Assignment lists length and seeds number do not match\n'
                                            f'Assignment lists lengths are: {len(self.fixed_idx)}, {len(self.warped_moving_idx)}'
                                            f'\nSeeds number are: {self.seeds_tips_fixed.shape[-1]},'
                                            f' {self.seeds_tips_moving.shape[-1]} ')
                    else:
                        seeds1_assigned = self.seeds_tips_fixed[..., self.fixed_idx]
                        seeds2_assigned = self.seeds_tips_moving_warped[..., self.warped_moving_idx]
                        meta_fixed = self.fixed_dict['meta']
                        meta_moving = self.moving_dict['meta']
                        self.assignment_dists = np.array(
                            [calc_dist(seeds1_assigned[..., i], seeds2_assigned[..., i], meta_fixed, meta_moving, calc_max=True)
                             for i in range(seeds2_assigned.shape[-1])])
                        seeds1, seeds2, _, self.errors = analyze_distances(self.case_name, self.assignment_dists,
                                                                               seeds1_assigned, seeds2_assigned)

                        self.update_massage('Number of assignments - {}\n sum of distances - '
                                            '{:.2f}\n average distance - {:.2f}'
                                            .format(len(self.assignment_dists), sum(self.assignment_dists),
                                                    np.mean(self.assignment_dists)))
                        self.show_movement_buttons()
                else:
                    self.update_massage("Please assign pair first")
            if event == "-SHOW_MOVE-":
                self.plot_name = "moves"
                self.window['-IMAGE-'].update("./movement_output/{}/movements.png".format(self.case_name))
            if event == "-SHOW_PAIRS-":
                self.plot_name = "pairs"
                self.window['-IMAGE-'].update("./movement_output/{}/pairs.png".format(self.case_name))
            if event == "-SAVE-":
                try:
                    df = pd.DataFrame({self.df_features[0]:pd.to_datetime('today'), self.df_features[1]:self.case_name,
                                       self.df_features[2]:self.registration_type,
                                       self.df_features[3]:self.registration_params["optimizer"],self.df_features[4]:
                                           self.registration_params['metric'], self.df_features[5]:
                                       self.registration_params["iterations"], self.df_features[6]:self.registration_params['learning_rate'],
                                      self.df_features[7]:self.registration_params['accuracy'],
                                       self.df_features[8]:self.registration_params['convergence_val'],
                                       self.df_features[9]:np.nanmean(self.assignment_dists), self.df_features[10]:np.nanmedian(self.assignment_dists),
                                       self.df_features[11]:np.nanstd(self.assignment_dists),
                                       self.df_features[12]:np.nanmax(self.assignment_dists), self.df_features[13]:np.nanmean(self.errors)},
                                      index=[0])
                    df.to_csv("./results.csv", mode='a', index=False, header=False, encoding='utf-8')
                    self.update_massage("Experiment results were saved to results.csv")
                except PermissionError as e:
                    self.update_massage("You have no permission to edit the file results.csv while its open."
                                        " please close the file and try again")
                except TypeError as e:
                    self.update_massage("To save your results, you must register the dicoms, assign pairs and caluculate"
                                        "distances")





            # if event == '-MOVING_FOLDER-':
            #     print('moving folder', values['-MOVING_FOLDER-'])
            #     self.moving_dict = get_all_dicom_files(values['-MOVING_FOLDER-'], {})
            #     self.moving_array = read_dicom(self.moving_dict['CT'])
            #     self.moving_viewer = DicomCanvas(self.moving_array, self.window['-MOVING_CANVAS-'].TKCanvas)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

    def show_global_params(self):
        self.window['-OPT_TEXT-'].update(visible=True)
        self.window['-OPT_MENU-'].update(visible=True)
        self.window['-METRIC_TEXT-'].update(visible=True)
        self.window['-METRIC_MENU-'].update(visible=True)
        for i in range(1,4):
            self.window[f'-GLOBAL_PARAM_TEXT_{i}-'].update(visible=True)
            self.window[f'-GLOBAL_PARAM_{i}-'].update(visible=True)

    def clear_global_params(self):
        self.window['-OPT_TEXT-'].update(visible=False)
        self.window['-OPT_MENU-'].update(visible=False)
        self.window['-METRIC_TEXT-'].update(visible=False)
        self.window['-METRIC_MENU-'].update(visible=False)
        for i in range(1,4):
            self.window[f'-GLOBAL_PARAM_TEXT_{i}-'].update(visible=False)
            self.window[f'-GLOBAL_PARAM_{i}-'].update(visible=False)

    def clear_all_params(self):
        types = OPTIMIZERS
        self.clear_global_params()
        for t in types:
            self.window[f'-{t}_PARAM_TEXT_1-'].update(visible=False)
            self.window[f'-{t}_PARAM_1-'].update(visible=False)

    def show_relevant_params_input(self):
        if self.registration_type is not None:
            types = OPTIMIZERS
            self.clear_all_params()
            self.show_global_params()
            for t in types:
                self.window[f'-{t}_PARAM_TEXT_1-'].update(visible=True)
                self.window[f'-{t}_PARAM_1-'].update(visible=True)

        else:
            self.clear_all_params()

    def create_param_dicts(self, values):
        if self.registration_type is not None:
            self.registration_params['optimizer'] = values['-OPT_MENU-']
            self.registration_params['metric'] = values['-METRIC_MENU-']
            self.registration_params['sampling_percentage'] = values['-GLOBAL_PARAM_1-']
            self.registration_params['iterations'] = values['-GLOBAL_PARAM_2-']
            self.registration_params['convergence_val'] = values['-GLOBAL_PARAM_3-']
            self.registration_params['learning_rate'] = values['-GD_PARAM_1-']
            self.registration_params['accuracy'] = values['-LBFGS2_PARAM_1-']

    def create_empty_param_dict(self):
        self.registration_params['optimizer'] = None
        self.registration_params['metric'] = None
        self.registration_params['sampling_percentage'] = None
        self.registration_params['iterations'] = None
        self.registration_params['convergence_val'] = None
        self.registration_params['learning_rate'] = None
        self.registration_params['accuracy'] = None

    def show_assignment_uploader(self):
        self.window['-ASSIGN_TEXT-'].update(visible=True)
        self.window['-ASSIGN_INPUT-'].update(visible=True)
        self.window['-ASSIGN_BROWSER-'].update(visible=True)

    def show_movement_buttons(self):
        self.window['-SHOW_MOVE-'].update(visible=True)
        self.window['-SHOW_PAIRS-'].update(visible=True)
        self.window['-SAVE-'].update(visible=True)

    def plot_registration(self):
        plot_name = None
        if self.registration_type is None:
            png_files = self.find_files_with_extention(self.case_name, '.png')
            if len(png_files):
                for file in png_files:
                    plot_name = file[:-4]
                    if "Bspline" in plot_name:
                        break
                self.window['-IMAGE-'].update("./registration_output/{}/{}".format(self.case_name, file))
            return plot_name
        for file in os.listdir("./registration_output/{}".format(self.case_name)):
            if file == "{}.png".format(self.registration_type):
                self.window['-IMAGE-'].update("./registration_output/{}/{}".format(self.case_name, file))
                plot_name = file
                self.update_massage("Displaying metric plot for {} registration".format(plot_name))
                return plot_name

    def use_saved_transformation(self):
        if self.case_name in os.listdir('./registration_output'):
            tfm_path = None
            tfm_files = self.find_files_with_extention(self.case_name, '.tfm')
            if len(tfm_files) > 0:
                tfm_path = "./registration_output/{}/{}".format(self.case_name, tfm_files[0])
                for file in tfm_files:
                    if "bspline" in file:
                        tfm_path = "./registration_output/{}/{}".format(self.case_name, file)
                        break
            if tfm_path is None:
                self.update_massage(NO_REGISTRATION_ERR)
            else:
                self.update_massage("Using saved transformation file found at:\n{}".format(tfm_path))
                self.tfm = sitk.ReadTransform(tfm_path)
                fixed_sitk = sitk.Cast(sitk.GetImageFromArray(self.fixed_array), sitk.sitkFloat32)
                moving_sitk = sitk.Cast(sitk.GetImageFromArray(self.moving_array), sitk.sitkFloat32)
                self.warped_moving = sitk.GetArrayFromImage(warp_image(fixed_sitk, moving_sitk, self.tfm))
        else:
            self.update_massage(NO_REGISTRATION_ERR)

    def parse_lists_from_file(self, path):
        with open(path, 'r') as f:
            list1 = f.readline().strip().split(',')
            list2 = f.readline().strip().split(',')
        return list1, list2


if __name__ == "__main__":
    app = App()
    app.run()