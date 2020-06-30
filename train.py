from src.utils import Dataset, Data
from src.utils import AverageMeter
from src.utils import DiceLoss, Dice
import logging
import warnings
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
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
                 pretrained='imagenet', device='cuda:0', criterion=None):
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


def train_models(args=None):
    torch.manual_seed(20)
    data = Dataset(dataset_path=args.root_dir)
    dice = Dice()
    model = None
    for fold in range(args.folds):
        begin_train_model = time.time()
        train, val = data.get_pathes_train_val(fold)
        print(f'Train length = {len(train)},\nValidation length = {len(val)}')
        train_dataset = Data(train)
        val_dataset = Data(val)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        valid_loader = DataLoader(val_dataset, batch_size=1)

        val_dice_0 = 0; val_dice_1 = 0
        t = tqdm(total=args.epochs,
                 bar_format='{desc} | epoch = {postfix[0]}/' + str(args.epochs) + ' || ' +
                            '{postfix[1]} : {postfix[2]:>2.4f} | {postfix[3]} : {postfix[4]:>2.4f} | ' +
                            '{postfix[5]} : {postfix[6]:>2.4f} || {postfix[7]} : {postfix[8]:>2.4f} | ' +
                            '{postfix[9]} : {postfix[10]:>2.4f} | {postfix[11]} : {postfix[12]:>2.4f} | ',
                 postfix=[0, 'loss', 0, 'dice_bg', 0, 'dice_class', 0, 'val_loss', 0, 'val_dice_bg', val_dice_0,
                          'val_dice_class', val_dice_1],
                 desc='Train ' + args.model_name + ' on fold ' + str(fold + 1),
                 position=0, leave=True)
        if args.model_name == 'smp.Unet':
            model = smp.Unet(
                encoder_name=args.model,
                encoder_weights=args.pretrained,
                in_channels=args.in_channels,
                classes=args.classes,
                activation='sigmoid'
            ).to(args.device)
        loss = smp.utils.losses.DiceLoss()
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=args.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=args.device,
            verbose=True,
        )
        for epoch in range(0, args.epochs):
            begin_epoch_train = time.time()
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
        #     average_total_loss = AverageMeter()
        #     average_dice = [AverageMeter() for i in range(2)]
        #     torch.cuda.empty_cache()
        #
        #     model.train(True)
        #     t.postfix[0] = epoch + 1
        #     for data_train in train_loader:
        #         torch.cuda.empty_cache()
        #         inputs, masks = data_train
        #         inputs = inputs.to(args.device)
        #         masks = masks.to(args.device)
        #         optimizer.zero_grad()
        #         outputs = model(inputs).to(args.device)
        #         outputs = torch.where(outputs > 0.8, torch.ones(outputs.shape).to(args.device),
        #                                              torch.zeros(outputs.shape).to(args.device))
        #         outputs.requires_grad = True
        #         loss = args.criterion(outputs, masks)
        #         average_total_loss.update(loss.data.item())
        #         for i in range(2):
        #             average_dice[i].update(dice(outputs, masks, i).item())
        #         loss.backward()
        #         # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         #     scaled_loss.backward()
        #         optimizer.step()
        #         t.postfix[2] = average_total_loss.average()
        #         t.postfix[4] = average_dice[0].average()
        #         t.postfix[6] = average_dice[1].average()
        #         t.update(n=1)
        #
        #     # validation
        #     average_val_total_loss = AverageMeter()
        #     average_val_dice = [AverageMeter() for i in range(2)]
        #     model.eval()
        #     for data_val in valid_loader:
        #         inputs, masks = data_val
        #         inputs = inputs.to(args.device)
        #         masks = masks.to(args.device)
        #         outputs = model(inputs).to(args.device)
        #         loss = args.criterion(outputs, masks)
        #
        #         average_val_total_loss.update(loss.data.item())
        #         for i in range(2):
        #             average_val_dice[i].update(dice(outputs, masks, i).item())
        #
        #         t.postfix[8] = average_val_total_loss.average()
        #         t.postfix[10] = average_val_dice[0].average()
        #         t.postfix[12] = average_val_dice[1].average()
        #         t.update(n=0)
        #     val_loss = average_val_total_loss.average()
        #     val_dice_0 = average_val_dice[0].average()
        #     val_dice_1 = average_val_dice[1].average()
        #     scheduler.step((val_dice_0 + val_dice_1) / 2)
        # t.close()
        # end_model_train = time.time()
        # end_epoch_train = time.time()
        # time_train = end_model_train - begin_train_model
        # time_epoch_train = end_epoch_train - begin_epoch_train
        # print('Model time train = {}\nEpoch time train = {}'.format(time_train, time_epoch_train))
        torch.save(model, 'best_models.pth')
    return model


def predict_masks(args, model):
    torch.manual_seed(20)
    data = Dataset(dataset_path=args.root_dir)
    test = data.get_test_path()
    test_dataset = Data(test)
    test_loader = DataLoader(test_dataset, batch_size=1)
    model.eval()
    index = 0
    for data in test_loader:
        inputs, masks = data
        inputs = inputs.to(args.device)
        mask = masks.to(args.device)
        outputs = model(inputs).squeeze().to(args.device)
        outputs = torch.where(outputs > 0.5, torch.ones(outputs.shape).to(args.device),
                              torch.zeros(outputs.shape).to(args.device)).detach().cpu().numpy()
        outputs = (outputs * 255).astype('int')
        img_arr = np.zeros((512, 640, 3))
        for i in range(3):
            img_arr[:,:,i] = outputs
        head, tail = os.path.split(test[index][0])
        results_path = os.path.join(args.root_dir, 'results')
        if not os.path.exists(results_path):
            os.mkdir(results_path)
        cv2.imwrite(os.path.join(results_path, tail), img_arr)
        index += 1
        # if index == 10:
        #     break


def create_submission():
    sample_subm = pd.read_csv(os.path.join(args.root_dir, 'sample_submission.csv'))
    print(sample_subm.head())
    print(sample_subm.columns)
    df_sub = pd.DataFrame(columns = ['id', 'class', 'base64 encoded PNG (mask)'])
    res_path = os.path.join(args.root_dir, 'results')
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
    df.to_csv('submission_1.csv', index = False)

def read_submission():
    df = pd.read_csv('submission_1.csv')
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
        folds=1,
        epochs=10,
        classes=1,
        lr=1e-4,
        model='resnet34',
        model_name='smp.Unet',
        criterion=DiceLoss(),
        in_channels=3,
        device='cuda:' + str(current_device))
    # model = train_models(args=args)
    # torch.save(model, 'test_model.pth')
    # model = torch.load('best_models.pth')
    # predict_masks(args, model)
    create_submission()
    # read_submission()