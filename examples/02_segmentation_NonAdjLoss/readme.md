## Code organization

- datasets : converts input data into iterable datasets for training and inference.
  - `training.py` : build a dataset for inference.
  - `inference.py` : build the datasets for training and validation.
  - `mappings.py` : mapping of image labels.
- `architecture.py` : architecture of the CNN.
- `inference_canvas.py` : inference and metrics evaluation for the test dataset.
- `segment.py` : script for segmenting images based on a model.
- `train_segment.py` : training + segmentation of test dataset.
- `train.py` : training script.

The output folder will contain :

- figures plotting the various segmentation metrics and losses.
- logs of metrics and losses for each iteration on train and validation.
- `checkpoint.pth.tar` : last epoch model's parameters.
- `model_best_dice.pth.tar` : best performing model's parameters.

## About NonAdjLoss

[Semi-supervised Learning for Segmentation Under Semantic
Constraint](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_68),
Pierre-Antoine Ganaye, Michaël Sdika, Hugues Benoit-Cattin. MICCAI 2018

Image segmentation based on convolutional neural networks is proving to be a
powerful and efficient solution for medical applications. However, the lack of
annotated data, presence of artifacts and variability in appearance can still
result in inconsistencies during the inference. We choose to take advantage of
the invariant nature of anatomical structures, by enforcing a semantic constraint
to improve the robustness of the segmentation. The proposed solution is applied
on a brain structures segmentation task, where the output of the network is
constrained to satisfy a known adjacency graph of the brain regions. This
criteria is introduced during the training through an original penalization
loss named NonAdjLoss. With the help of a new metric, we show that the
proposed approach significantly reduces abnormalities produced during the
segmentation. Additionally, we demonstrate that our framework can be used
in a semi-supervised way, opening a path to better generalization to unseen data.

The implementation of the NonAdjLoss is contained in :

- `nonadjloss` folder :
  - `extract_adjacency_matrix.py` : extracts and sum adjacencies from ground truth
    segmentation maps.
  - `loss.py` : adjacency estimator and tuning of the lambda weighting parameter.

### How to

In practice this non-adjacency penalization is applied on a pre-trained model,
because it's simpler to optimize a model that's already good at segmenting structures.
Then enforcing this penalization can be seen as some kind of fine-tuning, where
first you train your cnn for segmentation and finally fine-tune it to produce
segmentations that are also correct with respect to the adjacency matrix. This
matrix can be extracted from the ground truth label maps, or given by hand.

In this example the adjacency matrix should be named `graph.csv` and located
in the train directory. Then your dataset should look something like this :

```bash
(pytorch-v1.0) [ganaye@iv-ms-593 miccai]$ ll
total 12
drwxrwxr-x 22 ganaye creatis 4096 Jan 26  2018 test
drwxrwxr-x 14 ganaye creatis 4096 Feb  2 18:41 train
drwxrwxr-x  7 ganaye creatis 4096 Jan 15  2018 validation

(pytorch-v1.0) [ganaye@iv-ms-593 miccai]$ ll train/
total 516
drwxrwxr-x 2 ganaye creatis   4096 Jan 23 11:09 1000
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1001
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1002
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1006
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1007
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1008
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1009
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1010
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1011
drwxrwxr-x 2 ganaye creatis   4096 Jan 12  2018 1012
-rw-rw-r-- 1 ganaye creatis      5 Jan 15  2018 allowed_data.txt
-rw-rw-r-- 1 ganaye creatis   1288 Jan 15  2018 class_log.csv
-rw-rw-r-- 1 ganaye creatis 455625 Jan 15  2018 graph.csv
-rw-rw-r-- 1 ganaye creatis   3030 Jan 15  2018 graph.png
-rw-rw-r-- 1 ganaye creatis     71 Jan 15  2018 stats_log.txt

(pytorch-v1.0) [ganaye@iv-ms-593 miccai]$ ll train/1000
total 41864
-rw-rw-r-- 1 ganaye creatis 16340661 Jan 15  2018 im_mni_bc.nii.gz
-rw-rw-r-- 1 ganaye creatis  8589615 Jan 15  2018 im_mni.nii.gz
-rw-rw-r-- 1 ganaye creatis      125 Jan 15  2018 mni_aff_transf.c3dmat
-rw-rw-r-- 1 ganaye creatis      190 Jan 15  2018 mni_aff_transf.mat
-rw-rw-r-- 1 ganaye creatis 16761108 Jan 15  2018 prepro_im_mni_bc.nii.gz
-rw-rw-r-- 1 ganaye creatis   344875 Jan 15  2018 prepro_seg_mni.nii.gz
-rw-rw-r-- 1 ganaye creatis   344193 Jan 15  2018 seg_mni.nii.gz

```

### General advice

Before training your first network with this penalization, we advise you to
compare the ground truth adjacencies to the ones produced by your baseline model.
In this way, you will have a first impression of how your solution performs and
if not, how much improvements you can expect from this work.

How to extract an adjacency matrix from an image, this example is part
of `extract_adjacency_matrix.py` :

```python
# adjacency matrix with one additional dimension for discarded label (-1)
img_adj = torch.FloatTensor(args.nb_labels + 1, args.nb_labels + 1).zero_()
# reads a segmentation map
label = SitkReader(train_patient + '/prepro_seg_mni.nii.gz')

# image array from the reader
label_array = label.to_torch().long()
# re-label discarded label (-1) by the last positive integer
label_array[label_array == -1] = args.nb_labels

# extract adjacency matrix from the image and fill in the matrix
image2graph3d_patch(label_array, img_adj.numpy(), args.nb_labels, args.n_size)
# discard last positive label (discarded label, you remember ?)
img_adj = img_adj[:-1, :-1]
```

### Training

To start training with a pre-trained model, use the following command :

```bash
python train_segment.py /mnt/hdd/datasets/processed/miccai /mnt/hdd/datasets/models/final/ -j 4 -b 2 --lr 0.001 --resume ~/model_best_dice.pth.tar
```

For more options :

```bash
(pytorch-v1.0) [ganaye@iv-ms-593 02_segmentation_NonAdjLoss]$ python train_segment.py --help
usage: train_segment.py [-h] [-j N] [--epochs N] [--start-epoch N] [-b N]
                        [--lr LR] [--momentum M] [--weight-decay W]
                        [--print-freq N] [--resume PATH]
                        DATA_DIR OUTPUT_DIR

PyTorch Automatic Segmentation Training and Inference

positional arguments:
  DATA_DIR              path to the data dir
  OUTPUT_DIR            path to the output directory (default: current dir)

optional arguments:
  -h, --help            show this help message and exit
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run (default: 200)
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 64)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.01)
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
```
