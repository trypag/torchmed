from itertools import product
import torch
from .sampler import Sampler


class MaskableSampler(Sampler):
    """Multi-processed sampler.

    Evaluates possible sampling positions with :func:`get_positions` and
    returns samples with :func:`__getitem__`. When :func:`Sampler.build` is called,
    a list of spatial coordinates is given to :func:`get_positions` for testing.

    An optional mask can be supplied to precise a sampling area.

    Parameters
    ----------
    pattern_map : dict
        mapping of the filename (key) with the corresponding pattern (value).
    nb_workers : int
        number of process used during sampling evaluation.
    offset : int, list[int]
        offset from one coordinate to the next one, equivalent to stride in CNNs.
        If the offset is an int, the same offset is used for all dimensions. If
        it's a list of int each dimension will get its own offset.


    .. note::
        the `pattern_map` keys should be prefixed by `image_` for inputs and
        `target_` for targets. The value associated to the key should be a tuple
        composed of: the input on which a pattern is applied and the pattern.

    """
    def __init__(self, pattern_map, offset=1, nb_workers=1):
        super(MaskableSampler, self).__init__(pattern_map, offset, nb_workers)

    def __getitem__(self, index):
        # Extract the data pattern, target pattern and the position of extract
        position = self._coordinates[index]

        result = []
        target = []
        for desc, composition in self._pattern_map.items():
            input_name, pattern = composition
            if desc.startswith('input'):
                if isinstance(input_name, list):
                    sub_result = []
                    for n in input_name:
                        sub_result.append(pattern(self._data[n].to_torch(), position))
                    result.append(sub_result)
                else:
                    result.append(pattern(self._data[input_name].to_torch(), position))
            elif desc.startswith('target'):
                target.append(pattern(self._data[input_name].to_torch(), position))

        result = result[0] if len(result) == 1 else result
        target = target[0] if len(target) == 1 else target

        if len(target) == 0:
            # if no pattern is specified for the target
            # extract the pixel value if it's available, otherwise None
            if 'target' in self._data and len(target) == 0:
                target = self._data['target'].to_torch()[tuple(position)]
            elif 'target' not in self._data:
                target = None

        # return the position of extraction with the extracted data & target
        return self._coordinates[index], result, target

    def get_positions(self, positions):
        """Evaluates valid sampling coordinates.

        For each position, check if the patterns are applyable, if yes, the
        position is added to the dataset. Allocates a numpy array of
        maximum possible size. This array is returned with the index of the
        last element, so that the caller can extract only the relevant part of
        this array.

        Parameters
        ----------
        positions : tuple(list[int], ..)
            tuple containing lists of int, each list for a dimension, each
            int for a coordinate.

        Returns
        -------
        tuple(:class:`torch.ShortTensor`, int)
            tensor of coordinates and the number of valid elements in it.

        """

        # if there is a mask use it
        if 'image_mask' in self._data.keys():
            img_array = self._data['image_mask'].to_torch()
            use_mask = True
        elif 'image_ref' in self._data.keys():
            img_array = self._data['image_ref'].to_torch()
            use_mask = False
        else:
            raise ValueError("data map must contain at least a reference image.")

        max_coord_nb = 1
        for n_coord in [len(l) for l in positions]:
            max_coord_nb *= n_coord
        coordinates = torch.ShortTensor(max_coord_nb, len(positions))

        index = 0
        for position in product(*positions):
            if not use_mask or (use_mask and img_array[position] == 1):

                # for each pixel, see if the patterns are applyable
                # if so, store the position for future extraction
                can_extract = []
                for desc, composition in self._pattern_map.items():
                    input_name, pattern = composition
                    if desc.startswith('input'):
                        can_extract.append(pattern.can_apply(
                            self._data[input_name].to_torch(), position))

                # if all of the patterns can be extracted
                if len(can_extract) > 0 and all(can_extract):
                    coordinates[index] = torch.ShortTensor(position)
                    index += 1

        return coordinates, index
