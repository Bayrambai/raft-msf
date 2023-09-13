#!/bin/bash

# DATASETS_HOME
KITTI_HOME="/data/datasets/deep_vision_group/kitt15_sf/"
CHECKPOINT="./ckpt/raft_msf.ckpt"

# model
MODEL=MonoSceneFlow_fullmodel

Valid_Dataset=KITTI_2015_Train_Full_mnsf
Valid_Augmentation=Augmentation_Resize_Only
Valid_Loss_Function=Eval_SceneFlow_KITTI_Train

# training configuration
SAVE_PATH="./raft-msf-eval"
python ../main.py \
--batch_size=1 \
--batch_size_val=1 \
--checkpoint=$CHECKPOINT \
--model=$MODEL \
--evaluation=True \
--num_workers=4 \
--save=$SAVE_PATH \
--start_epoch=1 \
--validation_augmentation=$Valid_Augmentation \
--validation_dataset=$Valid_Dataset \
--validation_dataset_preprocessing_crop=False \
--validation_dataset_root=$KITTI_HOME \
--validation_loss=$Valid_Loss_Function \
--validation_key=sf \
--iters 10 \
#--save_flow=True \
#--save_disp=True \
#--save_disp2=True
