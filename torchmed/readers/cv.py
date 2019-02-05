import cv2
import numpy as np
import torch

from .reader import Reader


class CvReader(Reader):
    """Image reader based on OpenCV.

    On call to :func:`to_torch` or :func:`to_numpy`, loads an image with openCV's
    :func:`cv2.imread` and returns a :class:`torch.Tensor`. The image is
    loaded on the first call and **kept in memory during the life of the reader**.

    A casting type for numpy and torch can optionally be specified, via
    `numpy_type` and `torch_type`. The casting is performed only once,
    during image reading.

    Args:
        path (string): a path to the image.
        transform (:class:`torch.Tensor` -> :class:`torch.Tensor`, optional):
            function to apply on the :class:`torch.Tensor` after image reading.
        numpy_type (:class:`numpy.dtype`, optional): numpy type to cast the
            :class:`numpy.ndarray` returned by :func:`cv2.imread`.
        torch_type (:class:`torch.dtype`, optional): torch type to cast the
            :class:`numpy.ndarray`.
        shared_memory (bool, optional): whether to move the underlying tensor
            storage into `shared memory
            <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.shared_memory_>`_.

    .. note::
        A modification in the tensor/array returned by :func:`to_torch` or
        :func:`to_numpy` will also affect the underlying storage. If you want
        a copy, with torch call :func:`.clone` and with numpy use :func:`.copy`.

    Examples:
        >>> x = torchmed.readers.CvReader("Screenshot.png")
        >>> x_arr = x.to_torch()
        >>> x_arr.size()
        torch.Size([928, 1421, 3])
        >>> # update the underlying storage
        >>> im_array[0, 0, 0] = 666
        >>> # save the modified array back to a file
        >>> im.to_image_file('test.png')
        >>> # loads the newly saved image
        >>> im2 = torchmed.readers.CvReader('test.png')
        >>> im2_array = im2.to_torch()
        >>> im_array.size() == im2_array.size()
        True
        >>> im2_array[0, 0, 0]
        tensor(666.)

    """
    def __init__(self, path, transform=None, numpy_type=None,
                 torch_type=None, shared_memory=True):
        super().__init__(path, transform, numpy_type, torch_type, shared_memory)

    def _torch_init(self):
        cv_image = cv2.imread(self._path)
        super()._torch_init(cv_image)

    def to_image_file(self, path, params=None):
        """Saves the underlying tensor back to an image

        Parameters
        ----------
        path : string
            a path to the output image.
        params : dict, optional
            a dictionnary of pairs (key, value) describing the parameters
            (see :func:`cv2.imwrite` function definition).

        Returns
        -------
        True or Exception
            Returns an Exception in case of failure by :func:`sitk.WriteImage`
            and True in case of success.

        """
        assert(isinstance(path, str))
        assert(len(path) > 0)

        if params is not None:
            assert(isinstance(params, dict))
            return cv2.imwrite(path, self.to_numpy(), **params)

        return cv2.imwrite(path, self.to_numpy())
