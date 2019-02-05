# TorchMed

Read and process medical images in PyTorch.

[![Build Status](https://travis-ci.com/trypag/pytorch-med.svg?token=W7UTQDqNUe21xtLfiqRm&branch=master)](https://travis-ci.com/trypag/pytorch-med)
[![codecov](https://codecov.io/gh/trypag/pytorch-med/branch/master/graph/badge.svg?token=kL3ASEka4B)](https://codecov.io/gh/trypag/pytorch-med)

---

This library is designed as a flexible tool to process various types N dimension images.
Through a set of image **readers** based on famous projects (SimpleITK, NiBabel, OpenCV, Pillow)
you will be able to load your data. Once loaded, specific sub-sampling of the original
data is performed with **patterns** (describing what/how to extract) and **samplers**
(checks where to extract).

With **readers**, **samplers** and **patterns**, you can compose **datasets** which
a perfectly suited for PyTorch.

## Install

From pip:

```bash
pip install torchmed
```

Locally :

```bash
python install setup.py
```

## Usage

### Reader

```python
>>> import torchmed

>>> image = torchmed.readers.SitkReader('prepro_im_mni_bc.nii.gz')
>>> label_map = torchmed.readers.SitkReader('prepro_seg_mni.nii.gz')
# gets image data
>>> image_array = image.to_torch()
>>> label_array = label_map.to_torch()

>>> image_array.size()
torch.Size([182, 218, 182])
>>> type(image_array)
<class 'torch.Tensor'>
>>> label_array[0,0,0]
tensor(0.)
# also available for Numpy
>>> type(image.to_numpy())
<class 'numpy.ndarray'>
```

### Pattern

Patterns are useful to specify how the data should be extracted from an image.
It is possible to apply several patterns on one or more images.

```python
>>> import torchmed

>>> image = torchmed.readers.SitkReader('prepro_im_mni_bc.nii.gz')
>>> square_patch = torchmed.patterns.SquaredSlidingWindow([182, 7, 182], use_padding=False)
# initialize the pattern with the image properties
>>> square_patch.prepare(image_arr)

# can_apply checks if a pattern can be applied at a given position
>>> square_patch.can_apply(image_arr, [0,0,0])
False
>>> square_patch.can_apply(image_arr, [91,4,91])
True
>>> square_patch.can_apply(image_arr, [91,3,91])
True
>>> square_patch.can_apply(image_arr, [91,2,91])
False
>>> square_patch.can_apply(image_arr, [91,154,91])
True

# to extract a patch at a correct position
>>> sample = square_patch(image_arr, [91,154,91])
>>> sample.size()
torch.Size([182, 7, 182])
```

### Sampler

Multi-processed sampler to automatically search for coordinates where sampling
(pattern extraction) is possible.

```python
>>> from torchmed.readers import SitkReader
>>> from torchmed.samplers import MaskableSampler
>>> from torchmed.patterns import SquaredSlidingWindow

# maps a name to each image
>>> file_map = {
...         'image_ref': SitkReader('prepro_im_mni_bc.nii.gz',
...             torch_type='torch.FloatTensor'),
...         'target': SitkReader('prepro_seg_mni.nii.gz',
...             torch_type='torch.LongTensor')
...     }

# sliding window pattern
>>> patch2d = SquaredSlidingWindow(patch_size=[182, 7, 182], use_padding=False)
# specify a pattern for each input image
>>> pattern_mapper = {'input': ('image_ref', patch2d),
...                   'target': ('target', patch2d)}
# muli-processed sampler with offset
>>> sampler = MaskableSampler(pattern_mapper, offset=[91, 1, 91], nb_workers=4)
>>> sampler.build(file_map)
>>> len(sampler)
212
>>> sample = sampler[0]
>>> type(sample)
<class 'tuple'>
>>> sample[0].size()
torch.Size([3])
>>> sample[1].size()
torch.Size([182, 7, 182])
>>> sample[2].size()
torch.Size([182, 7, 182])
```

### Dataset

`MedFile` and `MedFolder` are iterable datasets, returning samples from the input
data. Here is an example of how to build a `MedFolder` from a list of images.
A `MedFolder` takes as input a list of `MedFile`s.

```python
import os
from torchmed.datasets import MedFile, MedFolder

self.train_dataset = MedFolder(
        self.generate_medfiles(os.path.join(base_dir, 'train'), nb_workers))

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

```

### Examples

See the `datasets` folder of the examples for a more pratical use case.

#### Credits

Evaluation metrics are mostly based on MedPy.
