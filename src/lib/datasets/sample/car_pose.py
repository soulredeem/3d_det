from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
from PIL import ImageEnhance
from PIL import Image
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import logging

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]

    def __call__(self, img):
        out = img
        rand_num = np.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1
            out = transformer(out).enhance(r)
        return out

class CarPoseDataset(data.Dataset):
    def __init__(self):
        _transform_dict = {'brightness':0.5026, 'contrast':0.5935, 'sharpness':0.8386, 'color':0.5592}
        self.color = ColorJitter(_transform_dict)

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i: #
            i *= 2
        return border // i

    def read_calib(self, calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        mask_name = file_name
        img_path = os.path.join(self.img_dir, file_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        img = Image.fromarray(img)
        img = self.color(img)
        img = np.array(img)

        
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)#[width/2,  height/2]中心点
        s = max(img.shape[0], img.shape[1]) * 1.0 #max(width, height)
        flipped = False
        rot = 0
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s*1.0
                #s = s * np.random.choice(np.arange(1.0, 1.2, 0.2))#scale 裁剪系数
                w_border = self._get_border(128, img.shape[1])#图像宽 128
                h_border = self._get_border(128, img.shape[0])#图像高 128
                c[0] = int(img.shape[1]/2)
                c[1] = int(img.shape[0]/2)


                #c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border) #随机中心点x
                #c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) #随机中心点y
                
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                
            if np.random.random() < self.opt.aug_rot:

                rf = 10
                rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h])
    
        trans_output = get_affine_transform(c, s, rot, [self.opt.output_w, self.opt.output_h])
        trans_output_inv = get_affine_transform(c, s, rot, [self.opt.output_w, self.opt.output_h], inv=1)

        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             # (self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)
        #mask = cv2.warpAffine(mask, trans_output, 
        #                    (self.opt.output_w, self.opt.output_h),
        #                     flags=cv2.INTER_LINEAR)
#         tmp_img = inp.copy()
        inp = (inp.astype(np.float32) / 255.)
        #mask = mask.astype(np.float32)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        num_joints = self.num_joints# 关键点个数 9

        hm_hp = np.zeros((num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32)#heatmap human points
        kps = np.zeros((num_joints * 2), dtype=np.float32)#关键点，2D坐标num_joints = 9
        kps_cent = np.zeros((2), dtype=np.float32)#关键点，中心点2D坐标

        hp_offset = np.zeros((num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((num_joints), dtype=np.int64)
        hp_mask = np.zeros((num_joints), dtype=np.int64)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        ann = anns[0]
           
        cls_id = int(ann['category_id']) - 1
        pts = np.array(ann['keypoints'][:18], np.float32).reshape(num_joints, 2)

        kps_cent = pts[8]
        for j in range(num_joints):
            pts[j] = affine_transform(pts[j], trans_output)

        is_visible = True

        if pts[8, 0] < 0 or pts[8, 0] >= self.opt.output_w or pts[8, 1] < 0 or pts[8, 1] >= self.opt.output_h:
            is_visible = False

        if is_visible is True:
                
            h, w = self.opt.output_h, self.opt.output_w
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius))
            ct = np.array(
                [self.opt.output_w / 2.0, self.opt.output_h / 2.0], dtype=np.float32)#2D中心点
            ct_int = ct.astype(np.int32)
                

            hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            hp_radius = self.opt.hm_gauss \
                if self.opt.mse_loss else max(0, int(hp_radius))

            for j in range(num_joints):
                
#                 tmp_img = cv2.circle(tmp_img, (int(pts[j, 0] * self.opt.down_ratio), int(pts[j, 1] * self.opt.down_ratio)), 3, (255, 0, 0), 2)
                kps[j * 2: j * 2 + 2] = pts[j, :2] - ct_int#减去2D中心点
                     
                if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
                        pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h:
                         
                    pt_int = pts[j].astype(np.int32)
                    hp_offset[j] = pts[j] - pt_int
                    hp_ind[j] = pt_int[1] * self.opt.output_w + pt_int[0]
                    hp_mask[j] = 1
                        
                    draw_gaussian(hm_hp[j], pt_int, hp_radius)
                
        meta = {'file_name': file_name}
#         cv2.imwrite(img_path.split('/')[-1], tmp_img)
        ret = {'input': inp, 'hps': kps, 'kps_cent': kps_cent, 
               'opinv': trans_output_inv, 'meta': meta}
               
        if self.opt.hm_hp:
            ret.update({'hm_hp': hm_hp})
        if self.opt.reg_hp_offset:
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            meta = {'c': c, 's': s, 'img_id': img_id}
            ret['meta'] = meta
        return ret
