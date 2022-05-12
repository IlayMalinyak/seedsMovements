# __author:IlayK
# data:02/05/2022
import SimpleITK as sitk
import sys
import os
import numpy as np
import registration_gui as rgui
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def read_image(dir):
    """
    read image to ITK object
    :param dir: directory to read from
    :return: sitk.Image object
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image


def register(fixed_path, moving_path, out_path, type="bspline", params={}):
    """
    register two dicoms
    :param fixed_path: path to fixed dicom
    :param moving_path: path to moving dicom
    :param out_path: path to save transformation file
    :param type: type of registration (default is bspline)
    :param params:
    :return: parameter dict
    """
    # fixed = sitk.Cast(sitk.GetImageFromArray(fixed_path), sitk.sitkFloat32)
    # moving = sitk.Cast(sitk.GetImageFromArray(moving_path), sitk.sitkFloat32)
    fixed = read_image(fixed_path)
    moving = read_image(moving_path)
    # print("shapes itk ", fixed.GetHeight(), fixed.GetWidth(), fixed.GetDepth())

    print("----%s----" % type)
    if type == "Bspline":
    # if translation_first:
        outTx = bspline_nonrigid_registration(fixed, moving, out_path, params)
    elif type == "Rigid":
        outTx = euler_rigid_registration(fixed, moving, out_path, params)
    elif type == "similarity":
        outTx = similarity_rigid_registration(fixed, moving)
    else:
        raise ValueError("transformation of type: %s does not exist" %type)
    warped = warp_image(fixed, moving, outTx)
    file_name = type + ".tfm"
    sitk.WriteTransform(outTx, out_path)

    fixed_arr = np.transpose(sitk.GetArrayFromImage(fixed), (1,2,0))
    moving_arr = np.transpose(sitk.GetArrayFromImage(moving), (1,2,0))
    moving_warped_arr = np.transpose(sitk.GetArrayFromImage(warped), (1,2,0))

    return fixed_arr, moving_arr, moving_warped_arr, outTx


def warp_image(fixed, moving, outTx):
    """
    warped image according to transformation
    :param fixed: fixed image (sitk object)
    :param moving: moving image (sitk object)
    :param outTx: transformation
    :return: warped image (sitk object)
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(outTx)
    warped_img = resampler.Execute(moving)
    return warped_img


def bspline_nonrigid_registration(fixed_image, moving_image, out_path, params):
    """
    bspline deformable registration
    :param fixed_image: sitk.Image fixe image
    :param moving_image: sitk.Image moving image
    :param out_path: path to save transformation
    :param params: parameters dictionary
    :return: transformation
    """
    registration_method = sitk.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical
    # spacing we want for the finest resolution control grid.
    grid_physical_spacing = [50.0, 50.0, 50.0] # A control point every 5mm
    image_physical_size = [size*spacing for size,spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size/grid_spacing + 0.5) \
                 for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
    # The starting mesh size will be 1/4 of the original, it will be refined by
    # the multi-resolution framework.
    mesh_size = [int(sz/4 + 0.5) for sz in mesh_size]

    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    # Instead of the standard SetInitialTransform we use the BSpline specific method which also
    # accepts the scaleFactors parameter to refine the BSpline mesh. In this case we start with
    # the given mesh_size at the highest pyramid level then we double it in the next lower level and
    # in the full resolution image we use a mesh that is four times the original size.
    registration_method.SetInitialTransformAsBSpline(initial_transform,
                                                     inPlace=False,
                                                     scaleFactors=[1,2,4])

    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method = set_metric_and_optimizer(registration_method, params)
    registration_method.SetInterpolator(sitk.sitkBSpline)

    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    plot_metric(registration_method)

    final_transformation = registration_method.Execute(fixed_image, moving_image)
    exceute_metric_plot(registration_method, out_path, "Bspline")
    print('\nOptimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transformation


def euler_rigid_registration(fixed_image, moving_image, out_path, params):
    """
    3D rigid registration
    :param fixed_image: sitk.Image fixe image
    :param moving_image: sitk.Image moving image
    :param out_path: path to save transformation
    :param params: parameters dictionary
    :return: transformation
    """
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method = set_metric_and_optimizer(registration_method, params)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetInitialTransform(initial_transform)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    # registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    plot_metric(registration_method)


    final_transform = registration_method.Execute(fixed_image, moving_image)
    exceute_metric_plot(registration_method, out_path, "Rigid")


    # Always check the reason optimization terminated.
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform


def set_metric_and_optimizer(registration_method, params):
    """
    create metric and optimizer using params dictionary
    :param registration_method: sitk.RegistrationMethod object
    :param params: parameters dictionary. should have the keys -
    :return:
    """
    if params['metric'] == 'Mean Squares':
        registration_method.SetMetricAsMeanSquares()
    elif params['metric'] == 'Mutual information':
        registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetMetricSamplingPercentage(float(params['sampling_percentage']))
    if params['optimizer'] == 'LBFGS2':
        registration_method.SetOptimizerAsLBFGS2(solutionAccuracy=float(params['accuracy']),
                                                 numberOfIterations=int(params['iterations']),
                                                 deltaConvergenceTolerance=float(params['convergence_val']))
    elif params['optimizer'] == 'Gradient Decent':
        registration_method.SetOptimizerAsGradientDescent(learningRate=float(params['learning_rate']), numberOfIterations=int(params['iterations']),
                                                      convergenceMinimumValue=float(params['convergence_val']),
                                                          convergenceWindowSize=10)
    return registration_method


def exceute_metric_plot(registration_method, out_path, method_name):
    """
    create metric plot
    :param registration_method: sitk.RegistrationMethod object
    :param out_path: path for saving
    :param method_name: registration type
    :return: None
    """
    handles = []
    patch = mpatches.Patch(color='b', label='Multi Resolution Event')
    # handles is a list, so append manual patch
    handles.append(patch)
    # plot the legend
    plt.legend(handles=handles, loc='upper right')
    if method_name == "demons (non rigid)":
        plt.title("%s metric final results %.2f" % (method_name, registration_method.GetMetric()))
    else:
        plt.title("%s metric final results %.2f" % (method_name, registration_method.GetMetricValue()))
    plt.xlabel("iteration number")
    plt.ylabel("mean squares value (mm)")
    plt.savefig(f'{out_path[:-4]}.png')


def command_iteration(method):
    """
    command function to run each iteration
    :param method: sitk registration method object
    """
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():10.5f}")


def plot_metric(registration_method):
    """
    plot metric
    :param registration_method: sitk registration method object
    """
    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, rgui.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, rgui.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, rgui.update_multires_iterations)
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: rgui.plot_values(registration_method))
