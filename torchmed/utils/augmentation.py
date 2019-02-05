import cv2
import math
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import torch


def resize_2d(image, size, fx, fy, interp=cv2.INTER_CUBIC):
    """Resize an image to the desired size with a specific interpolator.

    image: (ndarray) 3d ndarray of size CxHxW.
    size: (tuple) output size of the image.
    interp: (int) OpenCV interpolation type.
    """
    image = image.numpy()
    res_img = cv2.resize(image.swapaxes(0, 2), size, fx, fy,
                         interpolation=interp).swapaxes(0, 2)
    return torch.from_numpy(res_img)


def center_rotate_2d(image, angle, scale=1., interp=cv2.INTER_NEAREST,
                     border_mode=cv2.BORDER_REPLICATE, border_value=0):
    """Apply a center rotation on a 2d image with a defined angle and scale.
    It supports 2d images with multiple channels.

    image: (ndarray) 3d ndarray of size CxHxW.
    angle: (float) rotation angle in degrees.
    scale: (float) scaling factor.
    interp: (int) OpenCV interpolation type.
    """
    image = image.numpy()
    h, w = image.shape[-2:]

    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((h // 2, w // 2), angle, scale)
    # apply the affine transform on the image
    res_img = cv2.warpAffine(image, rot_mat, (h, w),
                             flags=interp, borderMode=border_mode,
                             borderValue=border_value)
    return torch.from_numpy(res_img)


def elastic_deformation_2d(image, alpha, sigma, order=0, mode='constant',
                           constant=0, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_
    Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural
    Networks applied to Visual Document Analysis"
    Based on https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation

    image: (ndarray) 3d ndarray of size CxHxW.
    alpha: (number) Intensity of the deformation.
    sigma: (number) Sigma for smoothing the transformation.
    order: (int) coordinate remapping : order of the spline interpolation.
    mode: (str) coordinate remapping : interpolation type.
    constant: (int) constant value if mode is 'constant'.
    random_state: (RandomState) Numpy random state.
    """
    image = image.numpy()
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    # random displacement field
    def_x = random_state.rand(*shape[-2:]) * 2 - 1
    def_y = random_state.rand(*shape[-2:]) * 2 - 1

    # smooth the displacement field of x,y axis
    dx = cv2.GaussianBlur(def_x, (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(def_y, (0, 0), sigma) * alpha

    # repeat the displacement field for each channel
    dx = np.repeat(dx[np.newaxis, :], shape[0], axis=0)
    dy = np.repeat(dy[np.newaxis, :], shape[0], axis=0)

    # grid of coordinates
    x, z, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),
                          np.arange(shape[2]))

    indices = (z.reshape(-1, 1), np.reshape(x + dx, (-1, 1)),
               np.reshape(y + dy, (-1, 1)))

    def_img = map_coordinates(image, indices, order=order,
                              mode=mode).reshape(shape)
    return torch.from_numpy(def_img)


def max_crop_size_for_rotation(final_crop_size, max_rotation_angle):
    """
    In order to find L, the size of the largest square containing a rotated
    square of size final_crop_size, we need to decompose L into x and y.
    L = x + y
    x = sin(max_rotation_angle) * final_crop_size
    y = sin(180-90-max_rotation_angle) * final_crop_size
    """
    x = math.sin(math.radians(max_rotation_angle)) * final_crop_size
    y = math.sin(math.radians(180 - 90 - max_rotation_angle)) * final_crop_size

    return math.ceil(x + y)
