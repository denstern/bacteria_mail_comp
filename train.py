from src.utils import Dataset, Data
from src.utils import AverageMeter
from src.utils import DiceLoss, Dice, DiceBCELoss, TverskyLoss, FocalTverskyLoss
from src.utils import train_transform, dev_transform, train_seg_models
import logging
import warnings
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
from torch.nn import BCEWithLogitsLoss
import time
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import base64
import apex.amp as amp

warnings.filterwarnings('ignore')

logger = logging.getLogger('TRAIN_file')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('train.log', mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


class Args:
    def __init__(self, root_dir='../dataset_its', batch_size=8, folds=4, epochs=10,
                 lr=1e-3, img_size=(512,640), model='resnet34',
                 in_channels=1, classes=2, model_name='smp.Unet',
                 pretrained='imagenet', device='cuda:0', criterion=None,
                 apex=False):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.folds = folds
        self.epochs = epochs
        self.lr = lr
        self.img_size = img_size
        self.model = model
        self.in_channels = in_channels
        self.classes = classes
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = 'cuda:0'
        self.criterion = criterion
        self.apex = apex


def train_models(args=None):
    model = train_seg_models(args)
    return model


def create_submission(filename='file.csv', res_path='results'):
    sample_subm = pd.read_csv(os.path.join(args.root_dir, 'sample_submission.csv'))
    print(sample_subm.head())
    print(sample_subm.columns)
    df_sub = pd.DataFrame(columns = ['id', 'class', 'base64 encoded PNG (mask)'])
    res_path = os.path.join(args.root_dir, res_path)
    results_files = os.listdir(res_path)
    sub_list = []
    for ind, file in enumerate(results_files):
        # print(sample_subm.iloc[ind])
        file_name = os.path.join(res_path, file)
        with open(file_name, 'rb') as img:
            enc_string = base64.b64encode(img.read())
            sub_list.append(['{0:03}'.format(ind+1), sample_subm['class'].iloc[ind],
                            enc_string.decode('utf-8')])
    # print(np.array(sub_list))
    df = pd.DataFrame(sub_list, columns = ['id', 'class', 'base64 encoded PNG (mask)'])
    # df_sub = df_sub.append(sub_list, ignore_index=True)
    df.to_csv(filename, index = False)


def read_submission(filename = 'file.csv'):
    df = pd.read_csv(filename)
    img_enc = df[df.columns[2]].iloc[1]
    with open('tmp_bacteria.png', 'wb') as fp:
        fp.write(base64.b64decode(df[df.columns[2]].iloc[1].encode()))
    mask = cv2.imread('tmp_bacteria.png', 0)
    print()


if __name__ == '__main__':
    logger.info('Begin main function')
    current_device = torch.cuda.current_device()
    logger.info('Torch current device is = {}'.format(current_device))
    logger.info('Torch current device name is  = {}'.format(torch.cuda.get_device_name(current_device)))
    torch.cuda.empty_cache()
    args = Args(
        root_dir='./dataset_its/',
        batch_size=4,
        folds=4,
        epochs=30,
        classes=1,
        lr=1e-3,
        model='efficientnet-b1',
        model_name='smp.Unet',
        criterion=DiceLoss(),
        in_channels=3,
        device='cuda:' + str(current_device)
    )
    model = train_models(args=args)
    # model_name = 'model_{}_{}.pth'.format(args.model_name, args.model)
    # torch.save(model, model_name)
    # results_path = 'results_{}_{}'.format(args.model, args.model_name)
    # predict_masks(args, model, results_path)
    # create_submission(filename='submission_5.csv', res_path=results_path)
    # read_submission(filename='submission_3.csv')