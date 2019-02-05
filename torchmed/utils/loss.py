def dice_loss(output, target, ignore_index=None):
    """
    output : NxCxHxW Variable
    target :  NxHxW LongTensor
    ignore_index : int index to ignore from loss
    """
    encoded_target = output.detach() * 0
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

    numerator = 2 * (output * encoded_target).sum(0).sum(1).sum(1)
    denominator = output + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0
    denominator = denominator.sum(0).sum(1).sum(1)
    loss = 1 - (numerator / denominator)

    return loss.sum() / loss.size(0)
