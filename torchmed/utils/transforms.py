import numpy as np
import torch


class CenterSquareCrops(object):
    """Returns a set of crops at different resolutions.
    Args:
        resolutions (Iterable[int]): list of resolutions to extract
    Example:
        >>> CenterSquareCrops([25, 51, 75])
    """
    def __init__(self, resolutions):
        self.resolutions = resolutions

    def __call__(self, tensor):
        crops = []

        w, h = tensor.size()[-2:]
        for res in self.resolutions:
            x = int(round((w - res) / 2.))
            y = int(round((h - res) / 2.))
            crops.append(tensor[..., x:x + res, y:y + res])

        return crops


class Pad(object):
    def __init__(self, pad, mode='constant', fill_value=None):
        self.pad = pad
        self.mode = mode
        self.fill_value = fill_value

    def __call__(self, input):
        input = input.numpy()
        if self.mode == 'constant':
            x = np.pad(input, self.pad, self.mode,
                       constant_values=self.fill_value)
        else:
            x = np.pad(input, self.pad, self.mode)

        return torch.from_numpy(x)


class Unsqueeze(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, input):
        if isinstance(input, list):
            for i, item in enumerate(input):
                input[i] = item.unsqueeze(self.dimension)
            return input

        else:
            return input.unsqueeze(self.dimension)


class Squeeze(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, input):
        if isinstance(input, list):
            for i, item in enumerate(input):
                input[i] = item.squeeze(self.dimension)
            return input

        else:
            return input.squeeze(self.dimension)


class RemoveLastIndex(object):
    def __init__(self, dimension):
        self.dimension = dimension

    def __call__(self, input):
        if isinstance(input, list):
            for i, item in enumerate(input):
                input[i] = input[i].narrow(self.dimension,
                                           0,
                                           input[i].size(self.dimension) - 1)
            return input

        else:
            return input.narrow(self.dimension,
                                0,
                                input.size(self.dimension) - 1)


class TransformNFirst(object):
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, tensor_list):
        ret_list = []
        for i in range(0, len(tensor_list)):
            if i < self.n:
                ret_list.append(self.transform(tensor_list[i]))
            else:
                ret_list.append(tensor_list[i])

        return ret_list
