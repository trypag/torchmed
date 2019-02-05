import os
import pandas as pd
import random
import torch

from torchmed.datasets import MedFile, MedFolder
from torchmed.samplers import MaskableSampler
from torchmed.patterns import SquaredSlidingWindow
from torchmed.readers import SitkReader
from torchmed.utils.transforms import Pad
from torchmed.utils.augmentation import elastic_deformation_2d


class MICCAI2012Dataset(object):
    def __init__(self, base_dir, nb_workers):
        def transform_train(tensor):
            return tensor.permute(1, 0, 2)

        def transform_target(tensor):
            return tensor.permute(1, 0, 2)[3]

        self.train_dataset = MedFolder(
            self.generate_medfiles(os.path.join(base_dir, 'train'), nb_workers),
            transform=transform_train, target_transform=transform_target,
            paired_transform=self.elastic_transform)
        self.validation_dataset = MedFolder(
            self.generate_medfiles(os.path.join(base_dir, 'validation'), nb_workers),
            transform=transform_train, target_transform=transform_target,
            paired_transform=self.elastic_transform)

        # init all the images before multiprocessing
        for medfile in self.train_dataset._medfiles:
            medfile._sampler._coordinates.share_memory_()
            for k, v in medfile._sampler._data.items():
                v._torch_init()

        # init all the images before multiprocessing
        for medfile in self.validation_dataset._medfiles:
            medfile._sampler._coordinates.share_memory_()
            for k, v in medfile._sampler._data.items():
                v._torch_init()

        # read cumulated volume of each labels
        df = pd.read_csv(os.path.join(base_dir, 'train/class_log.csv'), sep=';', index_col=0)
        self.class_freq = torch.from_numpy(df['volume'].values).float()

    def generate_medfiles(self, dir, nb_workers):
        # database composed of dirname contained in the allowed_data.txt
        database = open(os.path.join(dir, 'allowed_data.txt'), 'r')
        patient_list = [line.rstrip('\n') for line in database]
        medfiles = []

        # builds a list of MedFiles, one for each folder
        for patient in patient_list:
            if patient:
                patient_dir = os.path.join(dir, patient)
                patient_data = self.build_patient_data_map(patient_dir)
                patient_file = MedFile(patient_data, self.build_sampler(nb_workers))
                medfiles.append(patient_file)

        return medfiles

    def build_patient_data_map(self, dir):
        # pads each dimension of the image on both sides.
        pad_reflect = Pad(((1, 1), (3, 3), (1, 1)), 'reflect')
        file_map = {
            'image_ref': SitkReader(
                os.path.join(dir, 'prepro_im_mni_bc.nii.gz'),
                torch_type='torch.FloatTensor', transform=pad_reflect),
            'target': SitkReader(
                os.path.join(dir, 'prepro_seg_mni.nii.gz'),
                torch_type='torch.LongTensor', transform=pad_reflect)
        }

        return file_map

    def build_sampler(self, nb_workers):
        # sliding window of size [184, 7, 184] without padding
        patch2d = SquaredSlidingWindow(patch_size=[184, 7, 184], use_padding=False)
        # pattern map links image id to a Sampler
        pattern_mapper = {'input': ('image_ref', patch2d),
                          'target': ('target', patch2d)}

        # add a fixed offset to make patch sampling faster (doesn't look for all positions)
        return MaskableSampler(pattern_mapper, offset=[92, 1, 92],
                               nb_workers=nb_workers)

    def elastic_transform(self, data, label):
        p = random.random()
        data_label = torch.cat([data, label.unsqueeze(0).float()], 0)

        # elastic deformation
        if p > 0.4:
            data_label = elastic_deformation_2d(
                data_label,
                data_label.shape[1] * 1.05,  # intensity of the deformation
                data_label.shape[1] * 0.05,  # smoothing of the deformation
                0,  # order of bspline interp
                mode='nearest')  # border mode

        data = data_label[0:7]
        label = data_label[7].long()

        return data, label
