import shutil
from pathlib import Path
import os

val_imgs_path = ('../data/AIC21_Track3_MTMC_Tracking/validation/*/*/img_with_bbox_nl/*.jpg')
val_imgs = [x for x in Path().glob(val_imgs_path)]

train_imgs_path = ('../data/AIC21_Track3_MTMC_Tracking/train/*/*/img_with_bbox_nl/*.jpg')
train_imgs = [x for x in Path().glob(train_imgs_path)]

all_imgs = val_imgs + train_imgs
for img in all_imgs:
    img.unlink()