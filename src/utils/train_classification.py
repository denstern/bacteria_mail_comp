from src.utils import DatasetCls, DataCls, AverageMeter
from src.utils import Dice, train_transform, dev_transform
import numpy as np
import pandas as pd
import os
import base64
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from matplotlib.image import imsave
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score
import torch.nn.functional as F

PATH = os.getcwd()
CLS_PATH = os.path.join(PATH, 'cls_results')
MODELS_PATH = ''
MODEL_RESULTS_PATH = ''


def create_folders(args):
    global MODELS_PATH, MODEL_RESULTS_PATH
    if not os.path.exists(CLS_PATH):
        os.mkdir(CLS_PATH)
    MODELS_PATH = os.path.join(CLS_PATH, args.model)
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)
    MODEL_RESULTS_PATH = os.path.join(MODELS_PATH, 'results')
    if not os.path.exists(MODEL_RESULTS_PATH):
        os.mkdir(MODEL_RESULTS_PATH)


def get_model_optim(args):
    model = None
    eff_models = ['efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3']
    if args.model in eff_models:
        model = EfficientNet.from_pretrained(args.model, num_classes=args.classes)
        model.to(args.device)
    else:
        raise Exception('Cannot create {} model'.format(args.model))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer


def train_model(args):
    torch.manual_seed(20)
    data = DatasetCls(dataset_path=args.root_dir)
    model = None
    for fold in range(args.folds):
        train, val = data.get_pathes_train_val(fold=fold)
        print(f'Train length = {len(train)},\nValidation length = {len(val)}')
        train_dataset = DataCls(train, transform=train_transform)
        val_dataset = DataCls(val, transform=dev_transform)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        valid_loader = DataLoader(val_dataset, batch_size=1)

        val_metric = 0
        t = tqdm(total=args.epochs,
                 bar_format='{desc} | epoch = {postfix[0]}/' + str(args.epochs) + ' || ' +
                            '{postfix[1]} : {postfix[2]:>2.4f} | {postfix[3]} : {postfix[4]:>2.4f} || ' +
                            '{postfix[5]} : {postfix[6]:>2.4f} | {postfix[7]} : {postfix[8]:>2.4f} |',
                 postfix=[0, 'loss', 0, 'train_avg_class', 0, 'val_loss', 0, 'val_avg_class', val_metric],
                 desc='Train ' + args.model + ' on fold ' + str(fold + 1),
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
        flag = True
        for epoch in range(0, args.epochs):
            average_total_loss = AverageMeter()
            average_metric = AverageMeter()
            torch.cuda.empty_cache()

            model.train(True)
            t.postfix[0] = epoch + 1
            for data_train in train_loader:
                torch.cuda.empty_cache()
                inputs, labels = data_train
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                optimizer.zero_grad()
                outputs = model(inputs).to(args.device)
                if flag:
                    flag = False
                # print("\n{}".format(outputs))
                loss = args.criterion(outputs, labels)
                average_total_loss.update(loss.data.item())
                values, indices = torch.max(outputs, dim=1)
                # print("\n{}".format(indices))
                outputs = F.one_hot(indices, num_classes=6)
                # outputs = torch.where(outputs == , torch.ones(result.shape).to(args.device),
                #                       #                       torch.zeros(result.shape).to(args.device)).detach().cpu().numpy()
                acc = f1_score(labels.data.to('cpu'), outputs.data.to('cpu'), average='samples')
                average_metric.update(acc)
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                optimizer.step()
                t.postfix[2] = average_total_loss.average()
                t.postfix[4] = average_metric.average()
                t.update(n=1)
            flag = True
            # validation
            average_val_total_loss = AverageMeter()
            average_val_metric = AverageMeter()
            model.eval()
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs = model(inputs).to(args.device)
                loss = args.criterion(outputs, labels)

                average_val_total_loss.update(loss.data.item())
                values, indices = torch.max(outputs, dim=1)
                outputs = F.one_hot(indices, num_classes=6)
                acc = f1_score(labels.data.to('cpu'), outputs.data.to('cpu'), average='samples')
                average_val_metric.update(acc)

                t.postfix[6] = average_val_total_loss.average()
                t.postfix[8] = average_val_metric.average()
                t.update(n=0)
            val_dice_0 = average_val_metric.average()
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


def ensamble_predict_cls(args):
    torch.manual_seed(20)
    data = DatasetCls(dataset_path=args.root_dir)
    test = data.get_test_path()
    test_dataset = DataCls(test, transform=dev_transform)
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


def make_prediction(args, labels):
    torch.manual_seed(20)
    data = DatasetCls(dataset_path=args.root_dir)
    test = data.get_test_path()
    test_dataset = DataCls(test, transform=dev_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    index = 0
    print('Ensemble models and make predictions...Please, wait')
    result_list = []
    model_list = []
    for fold in range(0, args.folds):
        model, _ = upload_model(args, fold)
        model_list.append(model)
    for inputs in test_loader:
        result = None
        for i in range(len(model_list)):
            # model, _ = upload_model(args, fold)
            model = model_list[i]
            model.eval()
            inputs = inputs.to(args.device)
            with torch.no_grad():
                outputs = model(inputs)
            if result is None:
                result = outputs
            else:
                result += outputs
        values, index = torch.max(result, dim=1)
        result = labels[index]
        result_list.append(result)
    return result_list


def create_submission(args, results, filename='file.csv'):
    # sample_subm = pd.read_csv(os.path.join(args.root_dir, 'sample_submission.csv'))
    previous_subm = pd.read_csv(os.path.join(os.getcwd(), filename))
    for i in range(len(results)):
        previous_subm['class'].iloc[i] = results[i]
    new_filename = filename[:-4] + '_ensemble_seg_cls.csv'
    previous_subm.to_csv(new_filename, index=False)


labels = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae',
          'staphylococcus_aureus', 'moraxella_catarrhalis',
          'c_kefir', 'ent_cloacae']


def train_cls_models(args):
    create_folders(args)
    # train_model(args)
    # model, _ = upload_model(args, fold=0)
    result_list = make_prediction(args, labels=labels)
    # ensamble_predict_cls(args)
    create_submission(args, results=result_list, filename='submission_7.csv')