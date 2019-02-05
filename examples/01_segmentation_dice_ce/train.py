import argparse
import os
import time
import torch

from torchmed.utils.logger_plotter import LoggerPlotter, MetricLogger
from torchmed.utils.loss import dice_loss

from architecture import ModSegNet
from datasets.mappings import ClassMapping
from datasets.training import MICCAI2012Dataset
from utils import *

parser = argparse.ArgumentParser(
    description='PyTorch Automatic Segmentation of brain MRI - Training')
parser.add_argument('data', metavar='DIR', help='path to the train dataset')
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
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--exp-id', default='willis', type=str, metavar='EXP_ID',
                    help='name of the experiment')


def main():
    global args
    args = parser.parse_args()

    #####
    #
    #                              Logging
    #
    #####
    log_plot = LoggerPlotter(args.output_dir,
                             ['train.csv', 'validation.csv',
                              'average_train.csv', 'average_validation.csv',
                              'learning_rate.csv'],
                             ['loss.png',
                              'average_dice.png', 'learning_rate.png',
                              'losses_train.png', 'losses_validation.png',
                              'average_iou.png'])

    log_metrics = ';'.join(['epoch', 'cross_entropy_loss', 'dice_loss',
                            'dice_metric', 'iou_metric'])
    log_plot.log('train.csv', log_metrics)
    log_plot.log('validation.csv', log_metrics)
    log_plot.log('average_train.csv', log_metrics)
    log_plot.log('average_validation.csv', log_metrics)
    log_plot.log('learning_rate.csv', ';'.join(['epoch', 'lr']))

    ce = MetricLogger('ce', 'CE Loss', '{:.2e}', '{:.3e}', '{:.5f}')
    dice_ls = MetricLogger('dice', 'Dice Loss', '{:.2e}', '{:.3e}', '{:.5f}')
    dice_met = MetricLogger('dice_metric', 'Dice', '{:>6.3f}', '{:>6.3f}', '{:.5f}')
    iou_met = MetricLogger('iou_metric', 'IoU', '{:>6.3f}', '{:>6.3f}', '{:.5f}')
    log_plot.add_metric([ce, dice_ls, dice_met, iou_met])

    #####
    #
    #                           Model loading
    #
    #####
    torch.backends.cudnn.benchmark = True
    nb_classes = ClassMapping().nb_classes
    model = ModSegNet(num_classes=nb_classes, n_init_features=7).cuda()

    # optionally resume from a checkpoint
    if args.resume is not None and args.resume != 'None':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #####
    #
    #                            Data loading
    #
    #####
    miccai12_dataset = MICCAI2012Dataset(args.data, nb_workers=args.workers)

    train_folder = miccai12_dataset.train_dataset
    train_loader = torch.utils.data.DataLoader(
        train_folder,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    val_folder = miccai12_dataset.validation_dataset
    val_loader = torch.utils.data.DataLoader(
        val_folder,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    #####
    #
    #                         Optimization / Loss
    #
    #####
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # median frequency balancing (useful for highly imbalanced problems)
    # Error Corrective Boosting https://arxiv.org/abs/1705.00938
    weights = miccai12_dataset.class_freq.median() / miccai12_dataset.class_freq
    criterion = torch.nn.NLLLoss(ignore_index=-1, weight=weights).cuda()

    best_dice = 0
    write_config(model, args, len(train_loader), len(val_loader))
    start_time = time.time()

    #####
    #                      Train / Validation Loop
    #####
    for epoch in range(args.start_epoch, args.epochs):
        # update learning rate with poly rate policy
        lr = poly_lr_scheduler(optimizer, args.lr, epoch, 1, args.epochs)

        train(train_loader, model, criterion, optimizer, epoch, log_plot)
        validate(val_loader, model, criterion, epoch, log_plot)

        val_dice = log_plot.metrics['dice_metric'].avg
        val_iou = log_plot.metrics['iou_metric'].avg

        # log learning rate and update figures
        log_plot.log('learning_rate.csv', '{:d};{:.7f}'.format(epoch, lr))
        update_figures(log_plot)

        # remember best prec@1 and save checkpoint
        is_best = val_dice > best_dice
        best_dice = max(val_dice, best_dice)
        save_checkpoint({
            'epoch': epoch,
            'experiment_id': args.exp_id,
            'state_dict': model.state_dict(),
            'dice_metric': val_dice,
            'iou_metric': val_iou
        }, is_best, args.output_dir)

    end_time = time.time()
    write_end_config(args, end_time - start_time)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    logger.clear_metrics()

    for i, (_, batch_img, target) in enumerate(train_loader):
        target_var = target.cuda()
        patch = batch_img.cuda()

        # compute output
        output = model(patch)
        ce = criterion(output, target_var)
        dice = 5 * dice_loss(output.exp(), target_var, -1)
        loss = ce + dice

        # compute gradient and do optim step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure dice metric
        indices = output.data.max(dim=1)[1].cpu().numpy()
        metrics_res = eval_metrics(indices, target.numpy())
        dice_sim, iou_metric = metrics_res
        logger.metrics['ce'].update(ce.data.item(), batch_img.size(0))
        logger.metrics['dice'].update(dice.data.item(), batch_img.size(0))
        logger.metrics['dice_metric'].update(dice_sim, batch_img.size(0))
        logger.metrics['iou_metric'].update(iou_metric, batch_img.size(0))

        if i % args.print_freq == 0:
            logger.print_metrics(epoch, i, len(train_loader))
            logger.write_val_metrics(epoch + (i / len(train_loader)), 'train.csv')

    logger.write_avg_metrics(epoch, 'average_train.csv')


def validate(val_loader, model, criterion, epoch, logger):
    logger.clear_metrics()

    with torch.no_grad():
        for i, (_, batch_img, target) in enumerate(val_loader):
            target_var = target.cuda()
            patch = batch_img.cuda()

            # compute output
            output = model(patch)
            ce = criterion(output, target_var)
            dice = 5 * dice_loss(output.exp(), target_var, -1)
            loss = ce + dice

            # measure dice metric
            indices = output.data.max(dim=1)[1].cpu().numpy()
            metrics_res = eval_metrics(indices, target.numpy())
            dice_sim, iou_metric = metrics_res
            logger.metrics['ce'].update(ce.data.item(), batch_img.size(0))
            logger.metrics['dice'].update(dice.data.item(), batch_img.size(0))
            logger.metrics['dice_metric'].update(dice_sim, batch_img.size(0))
            logger.metrics['iou_metric'].update(iou_metric, batch_img.size(0))

            if i % args.print_freq == 0:
                logger.print_metrics(epoch, i, len(val_loader), 'test')
                logger.write_val_metrics(epoch + (i / len(val_loader)),
                                         'validation.csv')

        logger.write_avg_metrics(epoch, 'average_validation.csv')


if __name__ == '__main__':
    main()
