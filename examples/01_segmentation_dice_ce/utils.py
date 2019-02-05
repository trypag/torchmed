import datetime
import numpy as np
import os
import shutil
import torch

from torchmed.utils.metric import dice as dc
from torchmed.utils.metric import jaccard, multiclass


def write_config(model, args, train_size, val_size):
    num_params = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            num_params += m.weight.data.numel()

    configfile = os.path.join(args.output_dir, 'config.txt')
    cfg_f = open(configfile, "a")
    cfg_f.write('\ntraining with {} patches\n'
                'validating with {} patches\n'
                .format(train_size * args.batch_size,
                        val_size * args.batch_size))
    cfg_f.write(('project: {}\n' +
                 'number of workers: {}\n' +
                 'number of epochs: {}\n' +
                 'starting epoch: {}\n' +
                 'batch size: {}\n' +
                 'learning rate: {:.6f}\n' +
                 'momentum: {:.5f}\n' +
                 'weight-decay: {:.5f}\n' +
                 'number of parameters: {}\n')
                .format(args.exp_id, args.workers,
                        args.epochs, args.start_epoch,
                        args.batch_size, args.lr,
                        args.momentum, args.weight_decay, num_params)
                )
    cfg_f.write('\nstarted training at {}\n'.format(datetime.datetime.now()))
    cfg_f.flush()


def write_end_config(args, elapsed_time):
    configfile = os.path.join(args.output_dir, 'config.txt')
    cfg_f = open(configfile, "a")
    cfg_f.write('stopped training at {}\n'.format(datetime.datetime.now()))
    cfg_f.write('elapsed time : {:.2f} hours or {:.2f} days.'
                .format((elapsed_time) / (60 * 60),
                        (elapsed_time) / (60 * 60 * 24)))
    cfg_f.flush()


def update_figures(log_plot):
    # plot avg train loss_meter
    log_plot.add_line('cross_entropy', 'average_train.csv', 'epoch', 'cross_entropy_loss', "#1f77b4")
    log_plot.add_line('dice', 'average_train.csv', 'epoch', 'dice_loss', "#ff7f0e")
    log_plot.plot('losses_train.png', 'epoch', 'loss')

    # plot avg validation loss_meter
    log_plot.add_line('cross_entropy', 'average_validation.csv', 'epoch', 'cross_entropy_loss', "#1f77b4")
    log_plot.add_line('dice', 'average_validation.csv', 'epoch', 'dice_loss', "#ff7f0e")
    log_plot.plot('losses_validation.png', 'epoch', 'loss')

    # plot learning rate
    log_plot.add_line('learning_rate', 'learning_rate.csv',
                      'epoch', 'lr', '#1f77b4')
    log_plot.plot('learning_rate.png', 'epoch', 'learning rate')

    # plot dice
    log_plot.add_line('train', 'average_train.csv', 'epoch', 'dice_metric', '#1f77b4')
    log_plot.add_line('validation', 'average_validation.csv',
                      'epoch', 'dice_metric', '#ff7f0e')
    log_plot.plot('average_dice.png', 'epoch', 'dice', max_y=1)

    # plot iou
    log_plot.add_line('train', 'average_train.csv', 'epoch', 'iou_metric', '#1f77b4')
    log_plot.add_line('validation', 'average_validation.csv',
                      'epoch', 'iou_metric', '#ff7f0e')
    log_plot.plot('average_iou.png', 'epoch', 'iou', max_y=1)


def save_checkpoint(state, is_best, output_dir):
    filename = os.path.join(output_dir, 'checkpoint.pth.tar')
    bestfile = os.path.join(output_dir, 'best_log.txt')
    torch.save(state, filename)
    if is_best:
        bestfile_f = open(bestfile, "a")
        bestfile_f.write('epoch:{:>5d}  dice:{:>7.4f}  IoU:{:>7.4f}\n'.format(
            state['epoch'], state['dice_metric'], state['iou_metric']))
        bestfile_f.flush()
        shutil.copyfile(filename,
                        os.path.join(output_dir, 'model_best_dice.pth.tar'))


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr * (1 - iter / max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def eval_metrics(segmentation, reference):
    results, undec_labels = multiclass(segmentation, reference, [dc, jaccard])
    return list(map(lambda l: sum(l.values()) / len(l), results))
