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

from .data_interface import DInterface
from .coco_data import build as build_coco
import torch.utils.data
from .torchvision_datasets import CocoDetection


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    # if args.dataset_file == 'coco':
    return build_coco(image_set, args)
    # if args.dataset_file == 'coco_panoptic':
    #     # to avoid making panopticapi required for coco
    #     # from .coco_panoptic import build as build_coco_panoptic
    #     return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')