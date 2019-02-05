import numpy as np
import torch


class Reader(object):
    """Image reader parent class.

    On call to :func:`to_torch` or :func:`to_numpy`, loads an image and
    returns a :class:`torch.Tensor`. The image is loaded only once on the first
    call and **kept in memory during the life of the reader**.

    A casting type for numpy and torch can optionally be specified.

    Args:
        path (string): a path to the image.
        transform (:class:`torch.Tensor` -> :class:`torch.Tensor`, optional):
            a function to apply on the :class:`torch.Tensor` in :func:`_torch_init`.
        numpy_type (:class:`numpy.dtype`, optional): a numpy type to cast the
            :class:`numpy.ndarray` returned by reader.
        torch_type (:class:`torch.dtype`, optional): a torch type to cast the `numpy.ndarray`.
        shared_memory (bool, optional): whether to move the underlying tensor
            storage into `shared memory
            <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.shared_memory_>`_.
    """
    def __init__(self, path, transform, numpy_type, torch_type, shared_memory):
        assert(isinstance(path, str) and len(path) > 0)
        self._path = path
        self._transform = transform

        self._torch_tensor = None
        self._numpy_type = numpy_type
        self._torch_type = torch_type
        self._shared_memory = shared_memory

    def _torch_init(self, numpy_array):
        # cast to numpy type if necessary
        if self._numpy_type is not None:
            numpy_array = np.array(numpy_array, dtype=self._numpy_type)
        else:
            numpy_array = np.array(numpy_array)

        # numpy array of type int16 should be converted to int32,
        # because torch does not handles np.int16 type
        if numpy_array.dtype == np.int16:
            self._torch_tensor = torch.from_numpy(
                numpy_array.astype(np.int32))
        else:
            self._torch_tensor = torch.from_numpy(numpy_array)

        if self._torch_type is not None:
            self._torch_tensor = self._torch_tensor.type(self._torch_type)

        if self._transform is not None:
            self._torch_tensor = self._transform(self._torch_tensor)

        # Moves the underlying storage to shared memory.
        if self._shared_memory:
            self._torch_tensor.share_memory_()

    def to_numpy(self):
        """Returns a numpy array of the image.
        """
        if self._torch_tensor is None:
            self._torch_init()

        return self._torch_tensor.numpy()

    def to_torch(self):
        """Returns a torch tensor of the image.

        """
        if self._torch_tensor is None:
            self._torch_init()

        return self._torch_tensor
