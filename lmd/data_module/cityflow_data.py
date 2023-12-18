# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import data_module.transforms as trans

# car_type: wagon -> suv, couple -> sedan
# car_col: orange, purple, yellow is emmpty in data. however, keep it.
cityflow_feats = {'unknow_class': ['unknow'], 'car_type': ['bus', 'pickup', 'sedan', 'suv', 'truck', 'van'],
        'car_col' : ['red', 'blue', 'brown', 'gray', 'black', 'silver', 'green', 'white', 'yellow', 'orange', 'purple']}

def cityflow_build_voca(feats = cityflow_feats, n_coor_bins = 1000, n_wh_bins=1000, mergebin = True):
    """
    build vocabulary of tokens
    """
    voca = []

    # Add x,y coordinate tokens:
    coor_tokens = [f'coor_{i}' for i in range(n_coor_bins)]
    voca.extend(coor_tokens)

    if not mergebin:
        # Add w,h coordinate tokens:
        wh_tokens = [f'wh_{i}' for i in range(n_wh_bins)]
        voca.extend(wh_tokens)

    # unknow is the first tokens of feats
    voca.extend(['unknown'])

    # Add feats token
    for _, v in feats.items():
        voca.extend(v)

    # Add unknown token for unknown feat

    return voca


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file, cache_mode=cache_mode)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # This prepare is used for segmentation only, do not need to care
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        # Target here contain {'image_id': image_id, 'annotations': target}
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return trans.Compose([
            trans.RandomHorizontalFlip(),
            trans.RandomSelect(
                trans.RandomResize(scales, max_size=1333),
                trans.Compose([
                    trans.RandomResize([400, 500, 600]),
                    trans.RandomSizeCrop(384, 600),
                    trans.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return trans.Compose([
            trans.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.data_dir)
    assert root.exists(), f'===================provided cityflow path {root} does not exist====================='
    PATHS = {
        # "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "train": (root / "train", root / "mapper_nls2Track3_v1.json"),
        "val": (root / "validation", root / "mapper_nls2Track3_v1.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    # print (f'===================image_folder {img_folder} +++++++++=====================')
    # dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
    #                         local_rank=get_local_rank(), local_size=get_local_size())
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set))
    return dataset
