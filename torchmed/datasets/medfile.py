from torch.utils.data import Dataset


class MedFile(Dataset):
    """Storage and utility class to process an image. Takes some input data,
    a way to sub-sample it and access to the samples.

    Parameters
    ----------
    data_map : dict
        a dictionnary that maps a key referencing an image, to a value being
        one of the :mod:`torchmed.readers`. `image_ref` is a mandatory key
        referencing the main image to be read.
    sampler : :class:`torchmed.Sampler`
        a sampler defining how samples are extracted from the data (see
        :mod:`torchmed.samplers`).
    transform : Any -> Any
        function performing a transformation (ex: data augmentation)
        on the data returned by `sampler`.
    target_transform : Any -> Any
        function performing a transformation (ex: data augmentation)
        on the label (target) returned by `sampler`.
    paired_transform: Any -> Any
        function performing a transformation on the data tensor AND the
        label tensor. The transformation is applied after `transform` and
        `target_transform`.

    """
    def __init__(self, data_map, sampler, transform=None, target_transform=None,
                 paired_transform=None):
        self._sampler = sampler
        self._sampler.build(data_map)

        self._transform = transform
        self._target_transform = target_transform
        self._paired_transform = paired_transform

    def __getitem__(self, index):
        """Returns a sample at the corresponding index.

        Parameters
        ----------
        index : int
            index of the sample to get.

        Returns
        -------
        tuple(objects)
            tuple composed of : the spatial position of sampling, the sample,
            and the label if possible.

        """
        # Samples the data pattern, target pattern and the position of sampling
        position, data, label = self._sampler[index]

        # to apply a transform on the image and/or target
        if self._transform is not None:
            data = self._transform(data)

        if label is not None:
            if self._target_transform is not None:
                label = self._target_transform(label)

            if self._paired_transform is not None:
                data, label = self._paired_transform(data, label)

            return position, data, label
        else:
            return position, data

    def __len__(self):
        """Number of samples in the MedFile.
        """
        return len(self._sampler)
