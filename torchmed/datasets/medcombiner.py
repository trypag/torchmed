from torch.utils.data import Dataset


class MedCombiner(Dataset):
    """Combines a list of MedFolder/MedFile/Dataset into a single Dataset.

    Parameters
    ----------
    dataset_list : Iterable[:class:`torch.utils.data.Dataset`]
        iterable of Datasets to assemble and iterate over.
    get_composer : (Iterable[:class:`torch.utils.data.Dataset`], int) -> Any
        function returning an object based on Datasets and an index.
    len_composer : Iterable[:class:`torch.utils.data.Dataset`] -> int
        function returning the length of the dataset based on Datasets.

    """
    def __init__(self, dataset_list, get_composer, len_composer):
        self._datasets = dataset_list
        self._get = get_composer
        self._len = len_composer

    def __getitem__(self, index):
        """Gets at index, response is defined by get_composer.
        """
        return self._get(self._datasets, index)

    def __len__(self):
        """Number of samples that can be extracted, defined by len_composer
        on the MedFolders.
        """
        return self._len(self._datasets)
