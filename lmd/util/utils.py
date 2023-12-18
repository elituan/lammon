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

import os
from pathlib2 import Path
import cv2
import json
import numpy as np


COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        xywh = [int(x) for x in xywh]
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1)
        y2 = y1 + np.maximum(0., xywh[3] - 1)
        return (int(x1), int(y1), int(x2), int(y2))
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')

def draw_bboxes(img, img_path, mapper, offset=(0, 0)):
    """

    """
    key = str(img_path)
    key = str(Path(*img_path.parts[3:]))
    if key not in mapper.keys():
        return img

    for box,value in mapper[key].items():
        x1, y1, x2, y2 = xywh_to_xyxy(box.split('_'))
        # print(x1, y1, x2, y2)
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # box text and bar
        try:
            id = int(value[1])
            color = COLORS_10[id % len(COLORS_10)]
        except:
            id = value[1]
            color = COLORS_10[99999 % len(COLORS_10)]
        car_type = value[2]
        car_col = value[4]
        label = 'Veh{}_{}_{}'.format(id, car_type, car_col)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [0,0,0], 2)
    return img


def check_and_create(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    return folder_path

def get_img_path_ls(data_dir, img_dir = 'original_img', type ='jpg', sub_dirs = ['train', 'validation']):
    img_path_ls = []
    cameras = []
    root = Path(data_dir)
    modes = [root / sub_dir for sub_dir in sub_dirs]
    for mode in modes:
        for scenario in mode.iterdir():
            for camera in scenario.iterdir():
                cameras.append(camera)
                imgs = [x for x in camera.glob(f'{img_dir}/*.{type}')]
                img_path_ls.extend(imgs)
    return img_path_ls


def save_frame_from_video(args, output_dir ='new'):
    """ Extract image from videos
    Args:
          output_dir: dir_name in camera dir
          bbox: draw bbox
          text: write text on the top of bbox
    """

    with open(args.mapper, 'r') as f:
        mapper = json.load(f)

    src_img_paths = get_img_path_ls(data_dir=args.data_dir, type ='jpg', sub_dirs = ['train', 'validation'])

    #  Create destination dir
    des_dírs = list(set([img_path.parents[0] for img_path in src_img_paths]))
    for des_dir in des_dírs:
        str_path = str(des_dir).replace('train', f'train_{output_dir}')
        str_path = str_path.replace('validation', f'validation_{output_dir}')
        des_dir = Path(str_path)
        des_dir.mkdir(parents=True, exist_ok=True)
        print (f'++++++++++Create Cameras Dir {str(des_dir)}+++++++++++++++++++++++')

    for img_path in src_img_paths:
        str_path = str(img_path).replace('train', f'train_{output_dir}')
        str_path = str_path.replace('validation', f'validation_{output_dir}')
        output_img = Path(str_path)

        image = cv2.imread(str(img_path))
        image_out = draw_bboxes(image, img_path, mapper, offset=(0, 0))
        cv2.imwrite(str(output_img), image_out)
        print(f'Export img: {str(output_img)}')


def load_model_path(root=None, version=None, v_num=None, best=False):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif version is not None:
            return str(Path('./lightning_logs', version, 'checkpoints'))
        else:
            return str(Path('./lightning_logs', f'version_{v_num}', 'checkpoints'))

    if root==version==v_num==None:
        return None

    root = generate_root()
    if Path(root).is_file():
        return root
    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, version=args.load_ver, v_num=args.load_v_num)

def match_name_keywords(n, name_keywords):
    """
    Check if n is in name_keywords
    """
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def coco_metrics_name():
    metrics = [
        # "(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100",
        "AP@IoU_0.50_0.95",
        "(AP) @[ IoU=0.50      | area=   all | maxDets=100",
        "(AP) @[ IoU=0.75      | area=   all | maxDets=100",
        "(AP) @[ IoU=0.50:0.95 | area= small | maxDets=100",
        "(AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100",
        "(AP) @[ IoU=0.50:0.95 | area= large | maxDets=100",
        "(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1",
        "(AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10",
        "(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100",
        "(AR) @[ IoU=0.50:0.95 | area= small | maxDets=100",
        "(AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100",
        "(AR) @[ IoU=0.50:0.95 | area= large | maxDets=100",

    ]
    return metrics