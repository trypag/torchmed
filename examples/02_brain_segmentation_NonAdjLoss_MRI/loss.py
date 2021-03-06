import torch


class AdjacencyEstimator(torch.nn.Module):
    """Estimates the adjacency graph of labels based on probability maps.

    Parameters
    ----------
    nb_labels : int
        number of structures segmented.

    """
    def __init__(self, nb_labels):
        super(AdjacencyEstimator, self).__init__()

        # constant 2D convolution, needs constant weights and no gradient
        # apply the same convolution filter to all labels
        layer = torch.nn.Conv2d(in_channels=nb_labels, out_channels=nb_labels,
                                kernel_size=3, stride=1, padding=0,
                                bias=False, groups=nb_labels)
        layer.weight.data.fill_(0)

        canvas = torch.Tensor(3, 3).fill_(1)
        # fill 3x3 filters with ones
        for i in range(0, nb_labels):
            layer.weight.data[i, 0, :, :] = canvas

        # exlude parameters from the subgraph
        for param in layer.parameters():
            param.requires_grad = False

        self._conv_layer = layer
        # replicate padding to recover the same resolution after convolution
        self._pad_layer = torch.nn.ZeroPad2d(1)

    def forward(self, image):
        # padding of tensor of size batch x k x W x H
        p_tild = self._pad_layer(image)
        # apply constant convolution and normalize by size of kernel
        p_tild = self._conv_layer(p_tild) / 9

        # normalization factor
        norm_factor = image.size()[0] * image.size()[2] * image.size()[3]

        # old deprecated formulation
        # graph = torch.Tensor(image.size(1), image.size(1)).cuda()
        # graph.fill_(0)
        # for i in range(image.size(1)):
        #     p_tild_i = image[:, i, :, :].unsqueeze(1)
        #     # element product of image.exp() and p_tild. Sum over batch, W, H
        #     graph[:, i] = (image * p_tild_i).sum(0).sum(1).sum(1)

        # torch v1.0 einstein notation replaces following loop
        return torch.einsum('nihw,njhw->ij', image, p_tild) / norm_factor


class LambdaControl:
    """Utility to optimize the value of the lambda parameter, which is the
    weighting term of the NonAdjLoss.

    Parameters
    ----------
    graph_gt : :class:`torch.Tensor`
        ground truth adjacency map.
    tuning_epoch : int
        index of the epoch at which we stop increasing `lambda_`.
        For example : for a total training time of 200 epochs, tuning_epoch
        could be 180.
        To disable this tuning step, set the parameter to
        the total number of epoch (ex: 200 in the previous example).

    Attributes
    ----------
    nb_conn_ab_max : int
        Description of attribute `nb_conn_ab_max`.
    lambda_ : float
        weighting term of the NonAdjLoss.
    lambda_factor : float
        factor used to increase `lambda_` at each call to :func:`update`.
    train_nan_count : int
        counter of the number of nan values caught so far.
    last_good_epoch : int
        Index of the last epoch with a good seg metric.
        Good = decrease should not be more than 0.01 compared to `good_seg_metric_value`.
    good_seg_metric_value : float
        value of the segmentation metric at epoch 0. This is used as a reference
        point in order to monitor if the segmentation quality decreases during
        training. This is ultimately used to control `lambda_`.
    update_counter : int
        number of iterations without updating `lambda_`. `lambda_` is updated
        when `update_counter` reaches 5.
    graph_gt : :class:`torch.Tensor`
        ground truth adjacency map.
    tuning_epoch : int
        index of the epoch at which we stop increasing `lambda_`.
        For example : for a total training time of 200 epochs, tuning_epoch
        could be 180.
        To disable this tuning step, set the parameter to
        the total number of epoch (ex: 200 in the previous example).

    """
    def __init__(self, graph_gt, tuning_epoch):
        self.graph_gt = graph_gt
        self.nb_conn_ab_max = (graph_gt > 0).sum()
        self.lambda_ = 1
        self.lambda_factor = 1.3
        self.train_nan_count = 0
        self.last_good_epoch = 0
        self.good_seg_metric_value = None
        self.update_counter = 0
        self.tuning_epoch = tuning_epoch

    def get_config(self):
        if self.good_seg_metric_value is None:
            return (self.graph_gt, self.nb_conn_ab_max, self.lambda_, False)
        else:
            return (self.graph_gt, self.nb_conn_ab_max, self.lambda_, True)

    def update(self, epoch, avg_seg_loss_train, avg_nonadjloss_train,
               avg_seg_loss_val, has_train_nan):
        """Update the lambda parameter according to metrics from the previous epoch.

        Parameters
        ----------
        epoch : int
            index of the epoch.
        avg_seg_loss_train : float
            average training segmentation loss per image per epoch.
        avg_nonadjloss_train : float
            average training NonAdjLoss per image per epoch.
        avg_seg_loss_val : float
            average validation segmentation loss per image per epoch.
        has_train_nan : bool
            flag indicating if nan values have appeared during last training epoch.

        Returns
        -------
        int
            0 if training should continue.
            -1 or -2 to interrupt.

        """
        # Init lambda after first epoch
        if epoch == 0:
            self.lambda_ = 0.3 * (avg_seg_loss_train / avg_nonadjloss_train)
            self.good_seg_metric_value = avg_seg_loss_val
        elif epoch == self.tuning_epoch:
            self.update_counter = 4

        # if no issue was detected, check dice and update
        if has_train_nan is False and self.train_nan_count < 3:
            # automatic update of lambda
            if epoch < self.tuning_epoch:
                if self.good_seg_metric_value - avg_seg_loss_val >= 0.02:
                    print('--High decrease in dice detected, no lambda update for 5 epochs')
                    self.update_counter = 0
                elif self.good_seg_metric_value - avg_seg_loss_val >= 0.01:
                    print('--Small decrease in dice detected')

                elif epoch > 0:
                    if self.update_counter >= 4:
                        print('--Increase lambda')
                        self.lambda_ *= self.lambda_factor
                        self.update_counter = 0
                    else:
                        self.update_counter += 1

                    self.train_nan_count = 0
            else:
                dice_diff = self.good_seg_metric_value - avg_seg_loss_val
                if dice_diff > 0:
                    if self.update_counter >= 4:
                        print('--Decrease lambda to improve dice {:.4f}'.format(dice_diff))
                        self.lambda_ *= 0.9
                        self.update_counter = 0
                    else:
                        print('--Dice is not good enough {:.4f}'.format(dice_diff))
                        self.update_counter += 1
                else:
                    self.update_counter = 0

            if avg_seg_loss_val >= self.good_seg_metric_value - 0.01:
                print('--Logging as good epoch')
                self.last_good_epoch = epoch

        # if an error was detected decrease factor
        elif has_train_nan is True and self.train_nan_count < 3:
            print('--Decreasing lambda factor')
            self.lambda_ *= 0.9
            self.lambda_factor *= 0.98
            self.update_counter = 0
            self.train_nan_count += 1

        if epoch - self.last_good_epoch >= 5 and (epoch - self.last_good_epoch) % 5 == 0:
            if epoch - self.last_good_epoch >= 10:
                print('--Insufficient dice for 15 epochs, reboot')
                print('--Decreasing lambda factor')
                self.lambda_ *= 0.9
                return -2
            else:
                print('--Decreasing lambda because of deacreasing dice')
                self.lambda_ *= 0.95

            self.lambda_factor *= 0.98
            self.update_counter = 0

        if self.lambda_factor < 1:
            print('lambda_factor is < 1, ending.')
            return -1
        elif self.train_nan_count >= 3:
            print('--Too many errors, restart from a previous epoch')
            return -2
        else:
            return 0
