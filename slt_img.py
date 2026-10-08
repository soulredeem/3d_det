import glob
import os.path as osp
import shutil
imgs_slt = glob.glob(osp.join('/mnt/bd/motor-car-video/guoxiaoyang/dataset/' + 'super_kitti_nuscenes/image',
                                 '*.png'))

# with open('train_new.txt') as f1, open('train_super_kitti_nuscenes.txt','w+') as f2:
#     keywords = set(img_per.split('/')[-1] for img_per in imgs_slt)
#     for line in f1:
#         img_name = line.split(',')[0]
#         if img_name in keywords:
#             f2.writelines(line) 

# with open('train_super_kitti_nuscenes.txt','r') as f:
#     print(len(f.readlines()))

with open('val_new.txt') as f1, open('val_super_kitti_nuscenes.txt','w+') as f2:
    keywords = set(img_per.split('/')[-1] for img_per in imgs_slt)
    # print('007027_0.png' in keywords)
    # print(len(f1.readlines()))
    for line in f1:
        img_name = line.split(',')[0]
        if img_name in keywords:
            f2.writelines(line) 
with open('val_super_kitti_nuscenes.txt','r') as f:
    print(len(f.readlines()))

# with open('train_super_kitti_nuscenes.txt','r') as f1, open('val_super_kitti_nuscenes.txt','r') as f2:
#     imgs_train = set(img_per.split(',')[0] for img_per in f1)
#     imgs_val = set(img_per.split(',')[0] for img_per in f2)
#     print(len(imgs_train))
#     print(len(imgs_val))
#     # print(imgs_val)
#     # print('014953_0.png' in imgs_val)
#     imgs_all = imgs_train.union(imgs_val) 
#     # for img_per in imgs_slt:
#     #     img_name = img_per.split('/')[-1]
#     #     if img_name in imgs_all:
#     #         print(img_name)
#     #         break
#     with open('train_super_kitti_nuscenes.txt','r') as f1:
#         cnt = 0
#         for line in f1:
#             img_name = line.split(',')[0]
#             if img_name in imgs_val:
#                 cnt += 1
#                 print(img_name)
#         print(cnt)
