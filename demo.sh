#!/bin/bash

python3 src/demo.py --demo /opt/tiger/RTM3D_1226/demo_kitti_format/data/kitti/image \
       	--calib_dir /opt/tiger/RTM3D_1226/demo_kitti_format/data/kitti/calib \
	--load_model /opt/tiger/RTM3D_1226/kitti_format/exp/km3d_multi_class/model_last.pth \
       	--gpus 0 \
	--arch dlav0_46x_c
