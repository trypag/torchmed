import argparse
import datetime
import os
import random
import string
import time


parser = argparse.ArgumentParser(
    description='PyTorch Automatic Segmentation Training and Inference')
parser.add_argument('data', metavar='DATA_DIR', help='path to the data dir')
parser.add_argument('output_dir', default='', metavar='OUTPUT_DIR',
                    help='path to the output directory (default: current dir)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    args = parser.parse_args()

    code_dir = os.path.dirname(os.path.realpath(__file__))
    train_script = os.path.join(code_dir, 'train.py')
    inference_script = os.path.join(code_dir, 'segment.py')
    report_script_dir = os.path.join(code_dir, '../scripts/report/')
    report_scripts = ['plot_boxplot_labels_by_metric',
                      'plot_boxplot_patients_by_metric',
                      'write_patient_by_metric', 'write_label_by_metric']

    exp_id = os.path.basename(code_dir)
    now = datetime.datetime.now()
    dir_name = '{}_{}_{}@{}-{}_{}_{}'.format(now.year, now.month, now.day,
                                             now.hour, now.minute,
                                             exp_id, id_generator(2))
    output_dir = os.path.join(args.output_dir, dir_name)
    os.makedirs(output_dir)

    #####
    #
    #                         Training
    #
    #####
    print('--> The output folder is {}'.format(output_dir))
    print('--> Started train script')
    ret = os.system('python -u {} {} {} -j {} -b {} --epochs {} --lr {} --exp-id {} --resume {}'.format(
        train_script,
        args.data,
        output_dir,
        args.workers,
        args.batch_size,
        args.epochs,
        args.lr,
        exp_id,
        args.resume
    ))

    # in case training ended with an error
    if os.WEXITSTATUS(ret) != 0:
        return -1

    print('Sleeping for 5 seconds before segmentation. (read/write sync)')
    time.sleep(5)

    #####
    #
    #                       Segmentation
    #
    #####
    output_dir_seg = os.path.join(output_dir, 'segmentations')
    os.makedirs(output_dir_seg)

    segment_command = 'python -u {} {} {} {}'.format(
        inference_script,
        os.path.join(args.data, 'test'),
        os.path.join(output_dir, 'model_best_dice.pth.tar'),
        output_dir_seg
    )

    print(segment_command)
    os.system(segment_command)

    #####
    #
    #                        Reporting
    #
    #####
    output_dir_report = os.path.join(output_dir, 'reports')
    for r_script in report_scripts:
        report_command = 'python -u {} {} {}'.format(
            os.path.join(report_script_dir, r_script + '.py'),
            output_dir_seg,
            output_dir_report
        )
        os.system(report_command)
        time.sleep(1)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


if __name__ == '__main__':
    main()
