import torch
import torch.nn as nn
from src.utils import make_one_hot


class Dice(nn.Module):
    """
    Dice metric
    """
    def __init__(self, smooth=1.):
        super(Dice, self).__init__()
        self.smooth = smooth

    def forward(self, output, target, class_id=1):
        # num_cls = output.size()[1]
        # _, output = torch.max(output, 1, keepdim=True)
        # output = make_one_hot(output, classes=num_cls)
        # target = make_one_hot(target, classes=num_cls)
        # output_flat = output[:,class_id,:,:].contiguous().view(-1)
        # target_flat = target[:,class_id,:,:].contiguous().view(-1)
        # intersection = (output_flat * target_flat).sum()
        # value = (2. * intersection + self.smooth)/(output_flat.sum() + target_flat.sum() + self.smooth)
        # return value
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return (2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth)