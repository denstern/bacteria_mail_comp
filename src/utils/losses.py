import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_one_hot(labels, classes):
    print(labels.shape)
    try:
        one_hot = torch.cuda.FloatTensor(labels.size()[0],
                                         classes,
                                         labels.size()[2],
                                         labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
    except:
        target = one_hot.scatter_(1, labels.data.long(), 1)
    return target


class DiceLoss(smp.utils.base.Loss):
    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        # print(output.shape)
        # print(target.shape)
        # target = make_one_hot(target, classes=output.size()[1])
        # #         output = F.softmax(output, dim=1)
        # classes = [0]
        # loss = 0
        # for cls in classes:
        #     output_flat = output[:, cls, :, :].contiguous().view(-1)
        #     target_flat = target[:, cls, :, :].contiguous().view(-1)
        #     intersection = (output_flat * target_flat).sum()
        #     loss += 1 - ((2. * intersection + self.smooth) /
        #                  (output_flat.sum() + target_flat.sum() + self.smooth))
        iflat = output.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + self.smooth) / (tflat.sum() + iflat.sum() + self.smooth))


class TverskyLoss(smp.utils.base.Loss):
    def __init__(self, smooth=1e-7, alpha=0.7):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, output, target):
        iflat = output.view(-1)
        tflat = target.view(-1)
        true_pos = (iflat * tflat).sum()
        false_neg = ((1 - tflat) * iflat).sum()
        false_pos = (tflat * (1 - iflat)).sum()
        loss = 1 - ((true_pos + self.smooth) /
                    (true_pos + self.alpha + false_neg + (1 - self.alpha) * false_pos + self.smooth))
        return loss


class FocalTverskyLoss(smp.utils.base.Loss):
    def __init__(self, smooth=1e-7, alpha=0.7, gamma=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, output, target):
        iflat = output.view(-1)
        tflat = target.view(-1)
        true_pos = (iflat * tflat).sum()
        false_neg = ((1 - tflat) * iflat).sum()
        false_pos = (tflat * (1 - iflat)).sum()
        tversky_coef = ((true_pos + self.smooth) /
                       (true_pos + self.alpha + false_neg + (1 - self.alpha) * false_pos + self.smooth))
        loss = pow(1 - tversky_coef, self.gamma)
        return loss


def confusion_matris(output, target):
    tp = output * target
    fp = output * (1 - target)
    fn = (1 - output) * target
    return tp, fp, fn


class DiceBCELoss(smp.utils.base.Loss):
    def __init__(self, smooth=1e-7):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        iflat = output.view(-1)
        tflat = target.view(-1)
        tflat = tflat.type(torch.cuda.FloatTensor)
        intersection = (iflat * tflat).sum()
        dice_value = 1 - ((2. * intersection + self.smooth) / (tflat.sum() + iflat.sum() + self.smooth))
        ce_value = F.binary_cross_entropy(iflat, tflat, reduction='mean')
        loss = ce_value + dice_value
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # targets = torch.argmax(targets, 1)
        # inputs = torch.argmax(inputs, 1)
        if self.logits:
            CE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            # BCE_loss = F.binary_cross_entropy(inputs, targets)
            _, labels = targets.max(dim=1)
            CE_loss = nn.CrossEntropyLoss()(inputs, labels)

        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)