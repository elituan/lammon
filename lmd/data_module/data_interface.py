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


