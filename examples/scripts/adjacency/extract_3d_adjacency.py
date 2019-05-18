import argparse
import cv2
import matplotlib.pyplot
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchmed.readers import SitkReader

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', metavar='DIR_DATA',
                    help='path to the image dataset')
parser.add_argument('output_dir', default='', metavar='OUTPUT_DIR',
                    help='path to the output directory (default: current dir)')
parser.add_argument('nb_labels', default=21, metavar='N_LABELS', type=int,
                    help='number of labels in the dataset')
parser.add_argument('--n-size', default=1, type=int, metavar='SIZE',
                    help='size of the neighborhood')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def main():
    global args, nb_classes
    args = parser.parse_args()

    nb_classes = args.nb_labels
    image_dataset = ImageDataset(args.data_dir)

    print("=> building the train dataset")
    data_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    adj_mat = extract_from_data(data_loader)
    save2png(adj_mat, os.path.join(args.output_dir, 'adjacency_n_' + str(args.n_size)))


def extract_from_data(data_loader):
    nonAdjLoss_arr = torch.FloatTensor(nb_classes, nb_classes).zero_().cuda()
    adjacencyLayer = AdjacencyEstimator(nb_classes, args.n_size * 2 + 1).train().cuda()

    for i, target in enumerate(data_loader):
        print(str(i) + ' / ' + str(len(data_loader)), target.size())
        target_gpu = target.cuda()
        nonAdjLoss_arr += adjacencyLayer(target_gpu)

    return nonAdjLoss_arr.cpu()


class ImageDataset(Dataset):
    def __init__(self, base_dir):
        root_dir = os.path.join(base_dir, 'train')
        database = open(os.path.join(root_dir, 'allowed_data.txt'), 'r')
        patient_list = [line.rstrip('\n') for line in database]
        self.medfiles = []

        for patient in patient_list:
            if patient:
                patient_dir = os.path.join(root_dir, patient)
                r = SitkReader(os.path.join(patient_dir, 'prepro_seg.nii.gz'),
                               torch_type='torch.LongTensor')
                self.medfiles.append(r)

    def __len__(self):
        return len(self.medfiles)

    def __getitem__(self, idx):
        return self.medfiles[idx].to_torch()


class AdjacencyEstimator(torch.nn.Module):
    """Estimates the adjacency graph of labels based on probability maps.

    Parameters
    ----------
    nb_labels : int
        number of structures segmented.

    """
    def __init__(self, nb_labels, kernel_size=3):
        super(AdjacencyEstimator, self).__init__()

        # constant 3D convolution, needs constant weights and no gradient
        # apply the same convolution filter to all labels
        layer = torch.nn.Conv3d(in_channels=nb_labels, out_channels=nb_labels,
                                kernel_size=kernel_size, stride=1, padding=0,
                                bias=False, groups=nb_labels)
        layer.weight.data.fill_(0)

        canvas = torch.Tensor(kernel_size, kernel_size, kernel_size).fill_(1)
        # fill filters with ones
        for i in range(0, nb_labels):
            layer.weight.data[i, 0] = canvas

        # exlude parameters from the subgraph
        for param in layer.parameters():
            param.requires_grad = False

        self._conv_layer = layer
        # replicate padding to recover the same resolution after convolution
        self._pad_layer = torch.nn.ReplicationPad3d((kernel_size - 1) // 2)

    def forward(self, target):
        target_size = list(target.size())
        target_size.insert(1, self._conv_layer.in_channels)
        one_hot_size = torch.Size(target_size)

        image = torch.FloatTensor(one_hot_size).zero_().cuda()
        image.scatter_(1, target.unsqueeze(1), 1)
        del target

        # padding of tensor of size batch x k x W x H
        p_tild = self._pad_layer(image)
        # apply constant convolution and normalize by size of kernel
        p_tild = self._conv_layer(p_tild) / (self._conv_layer.kernel_size[0] ** 3)

        # normalization factor
        norm_factor = image.size()[0] * image.size()[2] * image.size()[3] * image.size()[4]

        return torch.einsum('nidhw,njdhw->ij', image, p_tild) / norm_factor


def save2png(image, output):
    # save to file
    np.savetxt(output + '.csv', image.float().numpy(), delimiter=';')

    # save binary image
    brain = (image > 0).float().numpy()
    brain = cv2.cvtColor(brain, cv2.COLOR_GRAY2BGR)
    brain = cv2.normalize(brain, brain, 0, 1, cv2.NORM_MINMAX)
    matplotlib.pyplot.imsave(output + '.png', brain)


if __name__ == '__main__':
    main()
