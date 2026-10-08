# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#g=$(($2<8?$2:8))

#srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
python3 ./src/main.py  --data_dir ./kitti_format --exp_id km3d_multi_class --arch dlav0_46x_c  --gpus 0 --batch_size 32 --lr 1.25e-3 --num_epochs 200 --save_all \
2>&1 | tee log.train
