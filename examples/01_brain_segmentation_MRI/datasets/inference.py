import os
import torch

from torchmed.datasets import MedFile
from torchmed.samplers import MaskableSampler
from torchmed.patterns import SquaredSlidingWindow
from torchmed.readers import SitkReader
from torchmed.utils.transforms import Pad


class MICCAI2012MedFile(object):
    def __init__(self, data, nb_workers):
        # a data map which specifies the image to read and reading specs
        # the transform pads each dimension of the image on both sides.
        patient_data = {
            'image_ref': SitkReader(
                os.path.join(data, 'prepro_im_mni_bc.nii.gz'),
                torch_type='torch.FloatTensor',
                transform=Pad(((1, 1), (0, 0), (1, 1)), 'reflect'))
        }
        # medfile dataset takes a data map, a Sampler and a transform
        self.test_data = MedFile(patient_data, self.buid_sampler(nb_workers),
                                 transform=lambda t: t.permute(1, 0, 2))

        # init all the images before multiprocessing
        self.test_data._sampler._coordinates.share_memory_()
        for k, v in self.test_data._sampler._data.items():
            v._torch_init()

    def buid_sampler(self, nb_workers):
        # sliding window of size [184, 1, 184] without padding
        patch2d = SquaredSlidingWindow(patch_size=[184, 1, 184], use_padding=False)
        # pattern map links image id to a Sampler
        pattern_mapper = {'input': ('image_ref', patch2d)}

        return MaskableSampler(pattern_mapper, nb_workers=nb_workers)
