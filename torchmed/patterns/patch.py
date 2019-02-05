import torch


class Pattern(object):
    """
    Abstracts a way to extract some informations from an image.
    `can_apply` returns True is the sample can be extracted at the given
    position of the image, and False otherwise. A `__call__` to the pattern
    will return the information of interest.
    Allows to write patterns of sampling independently and combine them flawlessly.
    """
    def prepare(self, image):
        raise NotImplementedError()

    def __call__(self, image, position):
        raise NotImplementedError()

    def can_apply(self, image, position):
        raise NotImplementedError()


class SquaredSlidingWindow(Pattern):
    """Square patch with padding sliding over the image (stride=1).

    Starts extraction from top left pixel to the bottom right. :func:`can_apply`
    tests if the patch can be extracted at the corresponding position.
    If `use_padding` is true, the latter condition will always be valid.
    First call :func:`__init__`, then :func:`prepare` to apply the pattern
    on the data.

    Parameters
    ----------
    patch_size : int or list
        size of the patch to extract.
    use_padding : bool
        whether to pad the image.
    pad_value : int
        padding fill value.

    """
    def __init__(self, patch_size, use_padding=False, pad_value=0):
        if isinstance(patch_size, (list, tuple)):
            assert(len(patch_size) > 0)
            assert(all(isinstance(n, int) for n in patch_size))
        else:
            assert(isinstance(patch_size, int))
        assert(isinstance(use_padding, bool))
        assert(isinstance(pad_value, int))

        self._patch_size = patch_size
        self._use_padding = use_padding
        self._pad_value = pad_value

    def prepare(self, image):
        """Initialize the pattern by matching the image properties to the
        pattern rules.

        Parameters
        ----------
        image : :class:`torch.Tensor`
            an N dimension array.

        """
        if isinstance(self._patch_size, int):
            self._patch_size = [self._patch_size] * image.ndimension()
        else:
            self._patch_size = list(self._patch_size)

        if any(x <= 0 for x in self._patch_size):
            raise ValueError('The patch size must be at least 1 '
                             'in any dimension.')
        elif len(self._patch_size) != image.ndimension():
            raise ValueError('The patch dimensionality must be equal '
                             'the image dimensionality.')

        self._pads = [(int(p / 2), int(p / 2 - (p - 1) % 2))
                      for p in self._patch_size]

    def can_apply(self, image, position):
        """Returns if the pattern can be applied at the given position.

        Parameters
        ----------
        image : :class:`torch.Tensor`
            image tensor.
        position : list[int]
            each axis coordinate.

        Returns
        -------
        bool
            True if the pattern can be applied at the given position,
            False otherwise.

        """
        if self._use_padding:
            return True
        else:
            is_size_ok = True
            for dim_size, pad_size, pos in zip(list(image.size()),
                                               self._pads, position):
                is_size_ok = dim_size - pad_size[1] > pos >= pad_size[0]
                if not is_size_ok:
                    is_size_ok = False
                    break
            return is_size_ok

    def __call__(self, image, position):
        """Get specified image pattern at given spatial position.

        Parameters
        ----------
        image : :class:`torch.Tensor`
            image tensor.
        position : list[int]
            list of spatial coordinates.

        Returns
        -------
        :class:`torch.tensor`
            extracted pattern from the image.

        """
        slices = ()
        for dim, p in enumerate(position):
            slices += (slice(p - self._pads[dim][0],
                             p + self._pads[dim][1] + 1),)

        # if no padding, narrow on both axis and return the view
        if not self._use_padding:
            return image[slices]
        else:
            # find the correct padding values
            pad_values = []
            for dim, s in enumerate(slices):
                crop_start = s.start
                crop_end = s.stop
                if crop_start < 0:
                    pad_values.append((dim, 0, abs(crop_start)))
                    crop_start = 0
                if crop_end > image.size(dim):
                    pad_values.append((dim, 1, crop_end - image.size(dim)))
                    crop_end = image.size(dim)

                image = image.narrow(dim, crop_start, crop_end - crop_start)

            # pad each dimension
            for dim, side, pad in pad_values:
                constant_s = []
                constant_e = []
                shape = list(image.size())
                shape[dim] = 1
                if side == 0:
                    constant_s = [torch.Tensor(*shape).type_as(image)
                                       .fill_(self._pad_value)] * pad
                else:
                    constant_e = [torch.Tensor(*shape).type_as(image)
                                       .fill_(self._pad_value)] * pad

                image = torch.cat(constant_s + [image] + constant_e, dim)

            return image
