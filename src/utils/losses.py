import segmentation_models_pytorch as smp
import torch


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
        iflat = output.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))