from src.utils import Dataset, Data, AverageMeter
from src.utils import Dice, train_transform, dev_transform
import numpy as np
import pandas as pd
import os
import base64
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from matplotlib.image import imsave

PATH = os.getcwd()
SEGM_PATH = os.path.join(PATH, 'segm_results')
MODELS_PATH = ''
MODEL_RESULTS_PATH = ''


def create_folders(args):
    global MODELS_PATH, MODEL_RESULTS_PATH
    if not os.path.exists(SEGM_PATH):
        os.mkdir(SEGM_PATH)
    MODELS_PATH = os.path.join(SEGM_PATH, args.model_name + '_' + args.model)
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    MODEL_RESULTS_PATH = os.path.join(MODELS_PATH, 'results')
    if not os.path.exists(MODEL_RESULTS_PATH):
        os.mkdir(MODEL_RESULTS_PATH)


def get_model_optim(args):
    model = None
    if args.model_name == 'smp.Unet':
        model = smp.Unet(
            encoder_name=args.model,
            encoder_weights=args.pretrained,
            in_channels=args.in_channels,
            classes=args.classes,
            activation='sigmoid'
        ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_model(args):
    torch.manual_seed(20)
    data = Dataset(dataset_path=args.root_dir)
    dice = Dice()
    model = None
    for fold in range(args.folds):
        train, val = data.get_pathes_train_val(fold=fold)
        print(f'Train length = {len(train)},\nValidation length = {len(val)}')
        train_dataset = Data(train, transform=train_transform)
        val_dataset = Data(val, transform=dev_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        valid_loader = DataLoader(val_dataset, batch_size=1)

        val_dice = 0
        t = tqdm(total=args.epochs,
                 bar_format='{desc} | epoch = {postfix[0]}/' + str(args.epochs) + ' || ' +
                            '{postfix[1]} : {postfix[2]:>2.4f} | {postfix[3]} : {postfix[4]:>2.4f} || ' +
                            '{postfix[5]} : {postfix[6]:>2.4f} | {postfix[7]} : {postfix[8]:>2.4f} |',
                 postfix=[0, 'loss', 0, 'dice_class', 0, 'val_loss', 0, 'val_dice_bg', val_dice],
                 desc='Train ' + args.model_name + ' on fold ' + str(fold + 1),
                 position=0, leave=True)
        model, optimizer = get_model_optim(args)
        # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True)
        best_model_checkpoint = {
            'epoch': 0,
            'accuracy': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dice': optimizer.state_dict(),
        }
        for epoch in range(0, args.epochs):
            average_total_loss = AverageMeter()
            average_dice = AverageMeter()
            torch.cuda.empty_cache()

            model.train(True)
            t.postfix[0] = epoch + 1
            for data_train in train_loader:
                torch.cuda.empty_cache()
                inputs, masks = data_train
                inputs = inputs.to(args.device)
                masks = masks.to(args.device)
                optimizer.zero_grad()
                outputs = model(inputs).to(args.device)
                loss = args.criterion(outputs, masks)
                average_total_loss.update(loss.data.item())
                average_dice.update(dice(outputs, masks, 0).item())
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                optimizer.step()
                t.postfix[2] = average_total_loss.average()
                t.postfix[4] = average_dice.average()
                t.update(n=1)

            # validation
            average_val_total_loss = AverageMeter()
            average_val_dice = AverageMeter()
            model.eval()
            for data_val in valid_loader:
                inputs, masks = data_val
                inputs = inputs.to(args.device)
                masks = masks.to(args.device)
                outputs = model(inputs).to(args.device)
                loss = args.criterion(outputs, masks)

                average_val_total_loss.update(loss.data.item())
                for i in range(2):
                    average_val_dice.update(dice(outputs, masks, i).item())

                t.postfix[6] = average_val_total_loss.average()
                t.postfix[8] = average_val_dice.average()
                t.update(n=0)
            val_dice_0 = average_val_dice.average()
            scheduler.step(val_dice_0)
            if val_dice_0 > best_model_checkpoint['accuracy']:
                best_model_checkpoint['epoch'] = epoch + 1
                best_model_checkpoint['model_state_dict'] = model.state_dict()
                best_model_checkpoint['optimizer_state_dice'] = optimizer.state_dict()
                best_model_checkpoint['accuracy'] = val_dice_0
            print('best model epoch = {} and accuracy is {}'.format(best_model_checkpoint['epoch'],
                                                                    best_model_checkpoint['accuracy']))
        t.close()
        model_name = "model_fold_{}.pth".format(fold)
        torch.save(best_model_checkpoint, os.path.join(MODELS_PATH, model_name))
    return model


def upload_model(args, fold):
    model, optimizer = get_model_optim(args)
    checkpoint = torch.load(os.path.join(MODELS_PATH, 'model_fold_{}.pth'.format(fold)))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dice'])
    return model, optimizer


def ensamble_predict_mask(args):
    torch.manual_seed(20)
    data = Dataset(dataset_path=args.root_dir)
    test = data.get_test_path()
    test_dataset = Data(test, transform=dev_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    index = 0
    print('Ensemble models and make predictions...Please, wait')
    for data in test_loader:
        result = None
        for fold in range(0, args.folds):
            model, _ = upload_model(args, fold)
            model.eval()
            inputs, masks = data
            inputs = inputs.to(args.device)
            outputs = model(inputs).squeeze().to(args.device)
            if result is None:
                result = outputs
            else:
                result += outputs
        result /= torch.tensor(args.folds).to(result.device)
        # outputs = torch.where(result > 0.8, torch.ones(result.shape).to(args.device),
        #                       torch.zeros(result.shape).to(args.device)).detach().cpu().numpy()
        outputs = result.round().int().cpu().numpy()
        outputs = (outputs * 255).astype(np.uint8)
        img_arr = np.zeros((512, 640, 3), dtype=np.uint8)
        for i in range(3):
            img_arr[:,:,i] = outputs
        head, tail = os.path.split(test[index][0])
        print('Image {} was saved'.format(tail))
        imsave(os.path.join(MODEL_RESULTS_PATH, tail), img_arr)
        index += 1


def create_submission(args, filename='file.csv'):
    sample_subm = pd.read_csv(os.path.join(args.root_dir, 'sample_submission.csv'))
    print(sample_subm.head())
    print(sample_subm.columns)
    results_files = os.listdir(MODEL_RESULTS_PATH)
    sub_list = []
    for ind, file in enumerate(results_files):
        # print(sample_subm.iloc[ind])
        file_name = os.path.join(MODEL_RESULTS_PATH, file)
        with open(file_name, 'rb') as img:
            enc_string = base64.b64encode(img.read())
            sub_list.append(['{0:03}'.format(ind + 1), sample_subm['class'].iloc[ind],
                             enc_string.decode('utf-8')])
    # print(np.array(sub_list))
    df = pd.DataFrame(sub_list, columns=['id', 'class', 'base64 encoded PNG (mask)'])
    # df_sub = df_sub.append(sub_list, ignore_index=True)
    df.to_csv(filename, index=False)


def train_seg_models(args):
    create_folders(args)
    train_model(args)
    ensamble_predict_mask(args)
    create_submission(args, filename='submission_7.csv')