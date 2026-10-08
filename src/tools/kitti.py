from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2

nuscenes = False

DATA_PATH = '/opt/tiger/RTM3D_1226/kitti_format/data/vvmotor/'
if nuscenes:
    DATA_PATH = './kitti_format/data/nuscenes/'
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import math

SPLITS = ['train1']
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image3, alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


def _bbox_to_coco_bbox(bbox):
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


def read_clib3(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 3:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


def read_clib0(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 0:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib


cats = ['Car']
det_cats = ['Car']
det_cats_ext = ['car']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_ids_ext = {cat: i + 1 for i, cat in enumerate(det_cats_ext)}

cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
    
    splits = ['train', 'val']
    if nuscenes:
        splits = ['train_nuscenes', 'val_nuscenes']

    calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                  'test': 'testing', 'train_stereo': 'training'}
    for split in splits:
        image_set_path = os.path.join(DATA_PATH, split, "new_image/")
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_set = open(DATA_PATH + '{}.txt'.format(split), 'r')
        image_to_id = {}
        for count, line in enumerate(image_set):
            if line[-1] == '\n':
                line = line[:-1]
            arr = line.split(',')
            file_name = arr[0].split('.')[0]
            image_id = int(file_name)
            keypoints = arr[1:]
            keypoints = [float(item) for item in keypoints]
            
            image_info = {'file_name': '{}.png'.format(file_name),
                          'id': int(image_id)}
            ret['images'].append(image_info)
            if split == 'test':
                continue

            ann = {'segmentation': [[0, 0, 0, 0, 0, 0]],
                    'area': 1,
                    'iscrowd': 0, 
                    'keypoints': keypoints,
                    'image_id': image_id,
                    'category_id': 1,
                    'id': int(len(ret['annotations']) + 1),
                   }
            ret['annotations'].append(ann)
        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        # import pdb; pdb.set_trace()
        out_path = '{}annotations/kitti_new_{}.json'.format(DATA_PATH, split)
        json.dump(ret, open(out_path, 'w'))
