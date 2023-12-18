# Create one face for each dataset or each setting.
# In each face, we load data_module, transform, augmentation at the same place

import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from pathlib import Path
import cv2

from torchvision import transforms
from sklearn.model_selection import train_test_split


class VitData(data.Dataset):
    def __init__(self, data_dir=r'../data/CIFAR10',
                 class_num=10,
                 train=True,
                 no_augment=True,
                 aug_prob=0.5,
                 img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.aug = train and not no_augment

        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value. 
        file_list_path = op.join(self.data_dir, 'file_list.pkl')
        with open(file_list_path, 'rb') as f:
            file_list = pkl.load(f)

        file_list = [self.data_dir / x for x in file_list]

        fl_train, fl_val = train_test_split(
            file_list, test_size=0.2, random_state=2333)
        self.path_list = fl_train if self.train else fl_val

        label_file = Path(self.data_dir) / 'label_dict.pkl'
        with open(label_file, 'rb') as f:
            self.label_dict = pkl.load(f)

    def __len__(self):
        return len(self.path_list)

    def to_one_hot(self, idx):
        out = np.zeros(self.class_num, dtype=float)
        out[idx] = 1
        return out

    def __getitem__(self, idx):
        path = self.path_list[idx]
        path_str = str(path)
        # print(path)
        # print(type(path))
        # img = np.load(path_str).transpose(1, 2, 0)
        img = cv2.imread(path_str)
        img = torch.from_numpy(img)

        labels = self.to_one_hot(self.label_dict[path.parts[-2]])
        labels = torch.from_numpy(labels).float()

        trans = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(self.aug_prob),
            transforms.RandomVerticalFlip(self.aug_prob),
            transforms.RandomRotation(10),
            transforms.RandomCrop(128),
            transforms.Normalize(self.img_mean, self.img_std)
        ) if self.train else torch.nn.Sequential(
            transforms.CenterCrop(128),
            transforms.Normalize(self.img_mean, self.img_std)
        )

        img_tensor = trans(img)

        return img_tensor, labels, path


# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .coco_data import build as build_coco_dataset
from .cityflow_data import build as build_cityflow_dataset
from util import misc
from util.utils import save_frame_from_video
# from vit_data import VitData
import torch.utils.data as torch_data


class DInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_train = None
        self.dataset_val = None
        self.sampler_train = None
        self.sampler_val = None
        self.save_hyperparameters()
        # In order to prevent any errors due to too many open file handles, try to reduce the number
        # of tensors to share, e.g., by stacking your data_module into a single tensor.
        # self.load_data_module()

    def prepare_data(self) -> None:
        if self.hparams.aic21_track3_extract:
            save_frame_from_video(args = self.hparams, output_dir = self.hparams.extract_dir)



    def setup(self, stage: str) -> None:
        # print(f'---------------------strategy {self.hparams.strategy}----------------------------------')
        # Build data
        # if stage == 'fit' or stage is None:
        if self.hparams.dataset_file == 'coco_data':
            self.dataset_train = build_coco_dataset(image_set='train', args=self.hparams)
            self.dataset_val = build_coco_dataset(image_set='val', args=self.hparams)
        elif self.hparams.dataset_file == 'cityflow_data':
            self.dataset_train = build_cityflow_dataset(image_set='train', args=self.hparams)
            self.dataset_val = build_cityflow_dataset(image_set='val', args=self.hparams)
        else:
            print(f'---------------------THERE IS NOT DATASET {self.hparams.dataset_file}----------------------------------')

        # Build Sampler
        number_using_data = self.hparams.number_using_data
        if number_using_data > 0:
            self.sampler_train = torch_data.RandomSampler(self.dataset_train, num_samples=number_using_data)
            self.sampler_val = torch_data.RandomSampler(self.dataset_val, num_samples=number_using_data)
        else:
            self.sampler_train = torch_data.RandomSampler(self.dataset_train)
            self.sampler_val = torch_data.RandomSampler(self.dataset_val)

    def train_dataloader(self):
        train_loader = DataLoader(self.dataset_train, batch_size=self.hparams.batch_size, sampler=self.sampler_train,
                          collate_fn=misc.collate_fn, num_workers=self.hparams.num_workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.hparams.batch_size, sampler=self.sampler_val,
                          collate_fn=misc.collate_fn, num_workers=self.hparams.num_workers, pin_memory=True)


