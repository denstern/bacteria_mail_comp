import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

p = 0.75
albu_train = A.Compose([
    # A.OneOf([
    #     A.HorizontalFlip(p=1),
    #     A.VerticalFlip(p=1),
    # ], p=p),

    # A.ToGray(p=0.5),

    # A.RandomBrightnessContrast(p=p),
    # A.OneOf([
    #     # A.ChannelShuffle(p=1),
    #     A.RGBShift(p=1),
    # ], p=0.5),

    # A.OneOf([
    #     A.RandomBrightnessContrast(p=1),
    #     A.RandomGamma(p=1)
    # ], p=p),
    # A.OneOf([
    #     A.Blur(p=1),
    #     A.MedianBlur(p=1),
    # ], p=p),
    # A.OneOf([
    #     A.RandomBrightnessContrast(p=1),
    #     A.RandomGamma(p=1),
    #     A.ChannelShuffle(p=0.2),
    #     A.HueSaturationValue(p=1),
    #     A.RGBShift(p=1),
    # ],p=p),
    ToTensorV2(),
])

albu_dev = A.Compose([
    ToTensorV2(),
])


def train_transform(img, mask):
    data = albu_train(image=img, mask=mask)
    img, mask = data['image'], data['mask']
    return img, mask.permute(2, 0, 1)


def dev_transform(img, mask):
    data = albu_dev(image=img, mask=mask)
    img, mask = data['image'], data['mask']

    return img, mask.permute(2, 0, 1)