import argparse
import torch

from architecture import ModSegNet
from inference_canvas import InferenceCanvas
from datasets.mappings import Miccai12Mapping
from datasets.inference import MICCAI2012MedFile


parser = argparse.ArgumentParser(
    description='PyTorch Automatic Segmentation (inference mode)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('model', metavar='MODEL',
                    help='path to a trained model')
parser.add_argument('output', metavar='OUTPUT',
                    help='path to the output segmentation map')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--wo-metrics', action='store_false',
                    help='whether to use metrics (dice, assd) or not')


def main():
    global args
    args = parser.parse_args()
    model = ModSegNet(num_classes=Miccai12Mapping().nb_classes,
                      n_init_features=1).cuda()
    inference_canvas = InferenceCanvas(args, infer_segmentation_map,
                                       MICCAI2012MedFile, model)
    inference_canvas()


def infer_segmentation_map(model, data_loader, label_map):
    probability_maps = []

    with torch.no_grad():
        for position, input in data_loader:
            output = model(input.cuda())
            _, predicted = output.data.max(1)

            # for each element of the batch
            for i in range(0, predicted.size(0)):
                y = position[i][1]
                label_map[:, y, :] = predicted[i].cpu().numpy()[1:-1, 1:-1]

    return probability_maps


if __name__ == '__main__':
    main()
