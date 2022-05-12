# __author:IlayK
# data:02/05/2022

# __author:IlayK
# data:28/04/2022
from matplotlib import  pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


class viewer():
    """
    class to view DICOMS
    """
    def __init__(self, ax, X, mask=None, aspect=1.0):
        self.ax = ax
        ax.set_title('Scroll to Navigate through the DICOM Image Slices')
        self.X = X
        self.mask = mask
        self.alpha = 0.4
        self.aspect = aspect
        if len(X.shape) == 3:
            rows, cols, self.slices = X.shape
            self.channels = 0
        else:
            rows, cols, self.channels, self.slices = X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[..., self.ind], cmap='gray')
        self.overlay = ax.imshow(self.mask[..., self.ind], cmap='Reds', interpolation='none',
                                 alpha=self.alpha) if self.mask is not None else None
        # self.ax.set_position([0.25, 0,1,1])
        if self.mask is not None:
            axAlph = plt.axes([0.2, 0.02, 0.65, 0.03])
            self.alpha_slider = Slider(
                ax=axAlph,
                label='Alpha',
                valmin=0,
                valmax=1,
                valinit=0.4,
            )
            self.alpha_slider.on_changed(self.update_alpha)
        ax.set_aspect(aspect)
        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def onkey(self, event):
        if event.key == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        self.update()

    def update_alpha(self, val):
        self.alpha = self.alpha_slider.val
        self.overlay.set_alpha(self.alpha)

    def update(self):
        self.im.set_data(self.X[..., self.ind])
        if self.mask is not None:
            self.overlay.set_data(self.mask[:, :, self.ind])
        self.ax.set_ylabel('Slice Number: %s' % self.ind)
        # cid = self.ax.canvas.mpl_connect('key_press_event', self.on_key)
        self.im.axes.figure.canvas.draw()


def show(arr, plane='axial'):
    """
    display axial plane interactively
    :param arr: array of data
    :param contours: array of contours
    :param seeds: array of seeds
    :param aspect: aspect ratio
    :param ax: matplotlib ax object (if None, a new one wil be created)
    """
    if plane == 'coronal':
        arr = np.swapaxes(arr, 0, 2)
    if plane == "sagittal":
        arr = np.swapaxes(arr, 0, 1)
    aspect = arr.shape[1]/arr.shape[0]
    fig, ax = plt.subplots(1, 1)
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = viewer(ax, arr, aspect=aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onkey)

    return plt.gcf()


def overlay_images(fixed_image, moving_image):
    """
    overlay two image in the same plot
    :param fixed_image: first image
    :param moving_image: second image
    """
    fig, ax = plt.subplots(1, 1)
    aspect = fixed_image.shape[1]/fixed_image.shape[0]
    # masked_seed_arr = np.ma.masked_where(seeds == 255, seeds) if seeds is not None else None
    tracker = viewer(ax, fixed_image, mask=moving_image, aspect=aspect)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('key_press_event', tracker.onkey)

    plt.show()


def display_slices(moving_image, fixed_image):
  """
  Displaying our two image tensors to register
  :param moving_image: [IM_SIZE_0, IM_SIZE_1, 3]
  :param fixed_image:  [IM_SIZE_0, IM_SIZE_1, 3]
  """
  # Display
  idx_slices = [int(5+x*5) for x in range(int(fixed_image.shape[3]/5)-1)]
  nIdx = len(idx_slices)
  plt.figure()
  for idx in range(len(idx_slices)):
      axs = plt.subplot(nIdx, 2, 2*idx+1)
      axs.imshow(moving_image[0,...,idx_slices[idx]], cmap='gray')
      axs.axis('off')
      axs = plt.subplot(nIdx, 2, 2*idx+2)
      axs.imshow(fixed_image[0,...,idx_slices[idx]], cmap='gray')
      axs.axis('off')
  plt.suptitle('Moving Image - Fixed Image', fontsize=200)
  plt.show()


def plot_individual_moves(case, dists, error):
    """
    plot individuals movements
    :param case: study id
    :param dists: array if distances
    :param error: array of errors (same length as knn_dists)
    :return: None
    """
    fig = plt.figure()
    # ax = fig.add_subplot()
    x = np.arange(1, len(dists) + 1)
    y = dists
    # plt.scatter(x, y, color="red")
    plt.errorbar(x, y, np.average(error, axis=0), fmt="o", color="b")

    plt.axhline(y=np.average(y), xmax=max(x), linestyle="--", label="Average", color="black")
    plt.title("%s individual moves" % case)
    plt.xlabel("# seed")
    plt.ylabel("movement (mm)")
    # plt.ylim((0,15))
    plt.legend()
    plt.savefig("./movement_output/%s/movements.png" % case)
    plt.close()


def plot_pairs(seeds1, seeds2, case):
    """
    plot seeds pairs
    :param seeds1: (3,3,N) array of first seeds. the first axis represent 3 tips (start, middle,end). the second axis represent
            3 coordinates (x,y,z)
    :param seeds2: (3,3,N) array of second seeds.
    :param case: case name
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n = seeds1.shape[-1]
    for i in range(n):
        ax.plot([seeds1[0,0, i], seeds1[2,0, i]], [seeds1[0,1, i], seeds1[2,1, i]], [seeds1[0,2, i], seeds1[2,2, i]], color='b')
        ax.plot([seeds2[0,0, i], seeds2[2,0, i]], [seeds2[0,1, i], seeds2[2,1, i]], [seeds2[0,2, i], seeds2[2,2, i]], color='orange')
        ax.plot([seeds1[1,0, i], seeds2[1,0, i]], [seeds1[1,1, i], seeds2[1,1, i]], [seeds1[1,2, i], seeds2[1,2, i]], color='green', alpha=0.5)
    plt.title(case)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig("./movement_output/%s/pairs.png" % (case))
    plt.close()


def display_dists(seeds1, seeds2, title, fname):
    """
    plot matchs and distances
    :param seeds1: 3XN array of first seeds
    :param seeds2: 3XN array of second seeds
    :param title: title for plot
    :param fname: file name for saving
    :return: None
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(seeds1.shape[-1]):
        if i == 0:
            ax.scatter(seeds1[0,i], seeds1[1,i], seeds1[2,i], color='b', label='postop')
            ax.scatter(seeds2[0,i], seeds2[1,i], seeds2[2,i], color='orange', label='removal')
        else:
            ax.scatter(seeds1[0, i], seeds1[1, i], seeds1[2, i], color='b')
            ax.scatter(seeds2[0, i], seeds2[1, i], seeds2[2, i], color='orange')
        ax.plot([seeds1[0, i], seeds2[0, i]], [seeds1[1, i], seeds2[1, i]], [seeds1[2, i],seeds2[2, i]], color='r')
    plt.legend()
    plt.title(title)
    plt.savefig("us/graphs/%s" % fname)
    plt.close()

