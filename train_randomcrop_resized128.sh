#!/bin/bash

# 加载环境
module load anaconda/2020.11
module load cuda/11.1
source activate SupCon

python main_supcon.py \
--batch_size 1024\
--model resnet50\
--size 128\
--learning_rate 0.1\
--temp 0.1\
--dataset path\
--data_folder ../lsz_kaggle_reorganized/train/image \
--method SimCLR  \
--mean "(0.4914, 0.4822, 0.4465)" \
--std "(0.5, 0.5, 0.5)" \
--trial train_kaggle_randomcrop_resized128