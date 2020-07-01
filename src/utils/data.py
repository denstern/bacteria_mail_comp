import os
import cv2
import random
import json
import logging
import numpy as np
import pandas as pd
from torch.utils import data
import torch
logger = logging.getLogger('TRAIN_file.DATA')


class Dataset:
    def __init__(self,
                 dataset_path='/home/denis/datasets/bacteria',
                 random_state=17,
                 folds=4,
                 ):
        np.random.seed(random_state)
        random.seed(random_state)
        self.train_path = os.path.join(dataset_path, 'train')
        self.train_mask_path = os.path.join(dataset_path, 'mask')
        self.test_path = os.path.join(dataset_path, 'test')

        self.df_info = pd.DataFrame()
        self.create_mask_images(self.train_mask_path, self.train_path)

        if folds:
            self.folds = max(0, folds)
            self.chunks = self.split_train(self.train_path)
        else:
            self.folds = None

    def create_mask_images(self, mask_path, train_path):
        files_info = []
        print(mask_path)
        if not os.path.exists(mask_path):
            os.mkdir(mask_path)
            logger.info('MASK path was created')
        files_json = [os.path.join(train_path, f) for f in os.listdir(train_path) if
                      f.endswith('.json')]
        for file in files_json:
            with open(file, 'r') as json_file:
                layout = json.load(json_file)
                h, w = layout['imageHeight'], layout['imageWidth']
                true_mask = np.zeros((h, w), np.uint8)
                label = layout['shapes'][0]['label']
                for shape in layout['shapes']:
                    polygon = np.array([point[::-1] for point in shape['points']])
                    cv2.fillPoly(true_mask, [polygon[:, [1, 0]]], 255)
                head, tail = os.path.split(file)
                cv2.imwrite(os.path.join(mask_path, tail[:-4] + 'png'), true_mask)
                files_info.append([tail, str(h), str(w), label])
        logger.info('All mask files were saved successfully saved to MASK path')
        self.df_info = pd.DataFrame(files_info, columns=['image_name', 'height', 'width', 'label'])

    def split_train(self, train_path):
        files_png = [os.path.join(train_path, f) for f in os.listdir(train_path) if
                     f.endswith('.png')]
        random.shuffle(files_png)
        chunks = np.array_split(files_png, 4)
        return chunks

    def get_pathes_train_val(self, fold):
        assert fold <= self.folds - 1
        if fold != -1:
            img_val_path = self.chunks[fold]
            mask_val_path = np.array([file.replace('train', 'mask') for file in img_val_path])
            img_train_path = np.concatenate(np.delete(self.chunks, fold))
            mask_train_path = np.array([file.replace('train', 'mask') for file in img_train_path])
            train = zip(img_train_path, mask_train_path, list([[512,640]] * len(img_train_path)))
            val = zip(img_val_path, mask_val_path, list([[512, 640]] * len(img_val_path)))
        else:
            img_train_path = np.concatenate(self.chunks)
            mask_train_path = np.array([file.replace('train', 'mask') for file in img_train_path])
            train = zip(img_train_path, mask_train_path, list([[512, 640]] * len(img_train_path)))
            val = []
        return list(train), list(val)

    def get_test_path(self):
        img_test_path = np.array([os.path.join(self.test_path, file) for file in os.listdir(self.test_path)])
        test = zip(img_test_path, img_test_path, list([[512,640]] * len(img_test_path)))
        return list(test)


class Data(data.Dataset):
    def __init__(self, pathes_shapes, transform, img_size=(256, 512)):
        self.pathes = pathes_shapes
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.pathes)

    def __getitem__(self, index):
        img = cv2.imread(self.pathes[index][0]).astype(np.float32)
        msk = (cv2.imread(self.pathes[index][1])).astype(np.int)
        img, msk = self.transform(img, msk)
        # img = torch.from_numpy(img.transpose(2, 0, 1))
        msk = (msk[0,:,:].unsqueeze(0) / 255).int()
        img = img / 255 - 0.5
        # msk = torch.from_numpy(msk.transpose(2, 0, 1))[0,:,:].unsqueeze(0)
        # shape = self.pathes[index][2]
        # img = cv2.resize(img.reshape((shape[0], shape[1], 3)), self.img_size, cv2.INTER_LINEAR) / 255  # - 0.5
        # msk = cv2.resize(msk.reshape((shape[0], shape[1], 1)), self.img_size, interpolation=0)
        # return np.expand_dims(img / 255 - 0.5, axis=0).astype(np.float32), \
        #        np.expand_dims(msk, axis = 2).astype(np.int)
        return img , msk