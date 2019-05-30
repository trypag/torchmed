import argparse
import numpy as np
import os

import cv2
import matplotlib.pyplot
import torch
from torch.utils.data import Dataset
from torchmed.readers import SitkReader


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', metavar='DIR_DATA',
                    help='path to the segmentation dataset')
parser.add_argument('output_dir', default='', metavar='OUTPUT_DIR',
                    help='path to the output directory (default: current dir)')
parser.add_argument('graph', metavar='FILE', help='path to GT graph')
parser.add_argument('--n-size', default=1, type=int, metavar='SIZE',
                    help='size of the neighborhood')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def main():
    global args, nb_classes
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_dataset = ImageDataset(args.data_dir)
    graph_master = np.loadtxt(args.graph, delimiter=';')
    nb_classes = graph_master.shape[0]

    print("=> building the train dataset")
    data_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    with torch.no_grad():
        mat = extract_from_data(data_loader, graph_master)
    save2png(mat, os.path.join(args.output_dir, 'adjacency_n_' + str(args.n_size)))


def extract_from_data(data_loader, graph_master):
    adjacencyLayer = AdjacencyEstimator(nb_classes, args.n_size * 2 + 1).train().cuda()
    spatialAdjacencyLayer = ContourEstimator(nb_classes, args.n_size * 2 + 1).train().cuda()
    adjacencyLayer.eval()
    spatialAdjacencyLayer.eval()

    log = open(os.path.join(args.output_dir, 'log.txt'), "w")
    log.write('id;unique;cumulated;m1;m2\n')
    m1_list = []
    m2_list = []

    for i, (p, target, p_name) in enumerate(data_loader):
        print(str(i) + ' / ' + str(len(data_loader)), target.size(), p)

        target_gpu = target.cuda()
        nonAdjLoss_arr = adjacencyLayer(target_gpu)
        contour_size = spatialAdjacencyLayer(target_gpu).cpu().numpy()[0]
        contour_size = (contour_size == 0).sum()

        nonAdjLoss_arr = nonAdjLoss_arr.cpu()
        nonAdjLoss_arr[torch.from_numpy(graph_master) >= 1] = 0

        m1 = (nonAdjLoss_arr > 0).sum().item() / (graph_master == 0).sum()
        m2 = 0
        if contour_size > 0:
            m2 = nonAdjLoss_arr.sum().item() / contour_size.item()

        m1_list.append(m1)
        m2_list.append(m2)

        log.write('{};{};{};{};{}\n'.format(p_name[0],
                                            (nonAdjLoss_arr > 0).sum(),
                                            nonAdjLoss_arr.sum(), m1, m2))
        log.flush()

    log.write('total;{} += {};{} += {}'.format(np.mean(m1_list),
                                               np.std(m1_list), np.mean(m2_list),
                                               np.std(m2_list)))
    log.flush()

    return nonAdjLoss_arr.cpu()


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
        # self._pad_layer = torch.nn.ReplicationPad3d((kernel_size - 1) // 2)
        self._pad_layer = torch.nn.ConstantPad3d((kernel_size - 1) // 2, 0)

    def forward(self, target):
        target_size = list(target.size())
        target_size.insert(1, self._conv_layer.in_channels)
        target_size = torch.Size(target_size)
        image = torch.FloatTensor(target_size).zero_().cuda()
        image.scatter_(1, target.unsqueeze(1), 1)

        # padding of tensor of size batch x k x W x H
        p_tild = self._pad_layer(image)
        # apply constant convolution and normalize by size of kernel
        p_tild = self._conv_layer(p_tild)

        return torch.einsum('nidhw,njdhw->ij', image, p_tild)


class ContourEstimator(torch.nn.Module):
    """Estimates the adjacency graph of labels based on probability maps.

    Parameters
    ----------
    nb_labels : int
        number of structures segmented.

    """
    def __init__(self, nb_labels, kernel_size=3):
        super(ContourEstimator, self).__init__()

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
        self._pad_layer = torch.nn.ConstantPad3d((kernel_size - 1) // 2, 0)

    def forward(self, target):
        target_size = list(target.size())
        target_size.insert(1, self._conv_layer.in_channels)
        target_size = torch.Size(target_size)
        image = torch.FloatTensor(target_size).zero_().cuda()
        image.scatter_(1, target.unsqueeze(1), 1)
        ret = torch.FloatTensor(target.size()).zero_().cuda()

        # padding of tensor of size batch x k x W x H
        p_tild = self._pad_layer(image)
        # apply constant convolution and normalize by size of kernel
        p_tild = self._conv_layer(p_tild) / 27.0

        for i in range(self._conv_layer.in_channels):
            for j in range(self._conv_layer.in_channels):
                if i != j:
                    ret[0, :, :, :] += p_tild[0, i] * image[0, j]
        return ret
        # return torch.einsum('nidhw,njdhw->dhw', image, p_tild)


def save2png(image, output):
    # save to file
    np.savetxt(output + '.csv', image.float().numpy(), delimiter=';')

    # save binary image
    brain = (image > 0).float().numpy()
    brain = cv2.cvtColor(brain, cv2.COLOR_GRAY2BGR)
    brain = cv2.normalize(brain, brain, 0, 1, cv2.NORM_MINMAX)
    matplotlib.pyplot.imsave(output + '.png', brain)


class ImageDataset(Dataset):
    def __init__(self, base_dir):
        self.medfiles = []

        for patient in sorted(os.listdir(base_dir)):
            p_dir = os.path.join(base_dir, patient)
            if os.path.exists(os.path.join(p_dir, 'segmentation.hdr')):
                seg_file = os.path.join(p_dir, 'segmentation.hdr')
            elif os.path.exists(os.path.join(p_dir, 'segmentation.nii.gz')):
                seg_file = os.path.join(p_dir, 'segmentation.nii.gz')
            else:
                raise Exception('Segmentation file does not exist (.nii.gz or .hdr)')

            if os.path.isdir(p_dir) and os.path.exists(seg_file):
                r = SitkReader(seg_file, torch_type='torch.LongTensor')
                self.medfiles.append((patient, r, patient))

    def __len__(self):
        return len(self.medfiles)

    def __getitem__(self, idx):
        return (self.medfiles[idx][0], self.medfiles[idx][1].to_torch(),
                self.medfiles[idx][2])


if __name__ == '__main__':
    main()
