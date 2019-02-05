from torch.utils.data import Dataset


class MedFolder(Dataset):
    """Packs a list of MedFiles into one dataset.
    Makes it easier to iterate over a list of MedFiles.

    Parameters
    ----------
    medfiles : Iterable[:class:`torchmed.datasets.MedFile`]
        list of MedFiles.
    transform : Any -> Any
        function performing a transformation (ex: data augmentation)
        on the data returned by a :class:`torchmed.Sampler`. This transform
        overwrites the default transform of the MedFiles.
    target_transform : Any -> Any
        function performing a transformation (ex: data augmentation)
        on the label (target) returned by a :class:`torchmed.Sampler`. This
        `target_transform` overwrites the default `target_transform` of the MedFiles.
    paired_transform: Any -> Any
        function performing a transformation on the data tensor AND the
        label tensor. The transformation is applied after `transform` and
        `target_transform`.This `paired_transform` overwrites the default
        `paired_transform` of the MedFiles.

    """
    def __init__(self, medfiles, transform=None, target_transform=None,
                 paired_transform=None):
        self._medfiles = medfiles
        self._map_file_index = []
        self._dataset_size = 0

        for index in range(0, len(self._medfiles)):
            self._medfiles[index]._transform = transform
            self._medfiles[index]._target_transform = target_transform
            self._medfiles[index]._paired_transform = paired_transform
            # dataset's size is increased by the size of the medfile
            self._dataset_size += len(self._medfiles[index])
            self._map_file_index.append((index, self._dataset_size))

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
            and the label if possible (see :func:`torchmed.datasets.MedFile.__getitem__`)

        """
        start_id = 0
        patient_id = 0
        for file_id, limit in self._map_file_index:
            if limit > index:
                patient_id = file_id
                start_ind = start_id
                break
            else:
                start_id = limit

        return self._medfiles[patient_id][index - start_ind]

    def __len__(self):
        """Number of samples in the MedFolder (sum of all the samples in the MedFiles).
        """
        return self._dataset_size
