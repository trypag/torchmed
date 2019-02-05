import numpy as np
import SimpleITK as sitk
import torch

from .reader import Reader


class SitkReader(Reader):
    """Image reader based on SimpleITK.

    On call to :func:`to_torch` or :func:`to_numpy`, loads an image with SimpleITK's
    :func:`sitk.ReadImage` and returns a :class:`torch.Tensor`. The image is
    loaded on the first call and **kept in memory during the life of the reader**.

    A casting type for numpy and torch can optionally be specified, via
    `numpy_type` and `torch_type`. The casting is performed only once,
    during image reading.

    Args:
        path (string): a path to the image.
        transform (:class:`torch.Tensor` -> :class:`torch.Tensor`, optional):
            function to apply on the :class:`torch.Tensor` after image reading.
        numpy_type (:class:`numpy.dtype`, optional): numpy type to cast the
            array returned by :func:`sitk.GetArrayFromImage`.
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
        >>> im = torchmed.readers.SitkReader("prepro_im_mni_bc.nii.gz")
        >>> im_array = im.to_torch()
        >>> im_array.size()
        torch.Size([182, 218, 182])
        >>> # update the underlying storage
        >>> im_array[0, 0, 0] = 666
        >>> # save the modified array back to a file
        >>> im.to_image_file('test_img.nii.gz')
        >>> # loads the newly saved image
        >>> im2 = torchmed.readers.SitkReader('test_img.nii.gz')
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
        itk_image = sitk.ReadImage(self._path)
        numpy_array = sitk.GetArrayFromImage(itk_image)
        super()._torch_init(numpy_array)

    def to_image_file(self, path, cast_type=None):
        """Saves the underlying tensor back to an image

        Parameters
        ----------
        path : string
            a path to the output image.
        cast_type : int, optional
            if you want to cast the pixel type before writing the image.

        Returns
        -------
        None or Exception
            Returns an Exception in case of failure by :func:`sitk.WriteImage`
            and None in case of success.

        """
        assert(isinstance(path, str))
        assert(len(path) > 0)

        if cast_type is not None:
            assert(isinstance(cast_type, int))

        itk_image = sitk.ReadImage(self._path)
        image = sitk.GetImageFromArray(self.to_numpy())
        image.CopyInformation(itk_image)

        if cast_type is None:
            itk_image = sitk.Cast(image,
                                  itk_image.GetPixelID())
        else:
            itk_image = sitk.Cast(image, cast_type)

        return sitk.WriteImage(itk_image, path)
