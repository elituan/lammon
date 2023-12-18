import shutil
from pathlib import Path
import os

train_imgs_path = ('../data/AIC21_Track3_MTMC_Tracking/train/*/*/original_img/*.jpg')
train_imgs = [x for x in Path().glob(train_imgs_path)]
val_imgs_path = ('../data/AIC21_Track3_MTMC_Tracking/validation/*/*/original_img/*.jpg')
val_imgs = [x for x in Path().glob(val_imgs_path)]
all_imgs = val_imgs + train_imgs

des_imgs =  [Path(str(path).replace('AIC21_Track3_MTMC_Tracking', 'aic22_original_frames')) for path in all_imgs ]
parent_dirs = list(set([img_path.parents[0] for img_path in des_imgs]))

# mkdir
for parent_dir in parent_dirs:
    parent_dir.mkdir(parents=True, exist_ok=True)

# Copy file
for i in range(len(all_imgs)):
    all_imgs[i].rename(des_imgs[i])