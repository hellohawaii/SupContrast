#!/bin/bash

# 加载环境
module load anaconda/2020.11
module load cuda/11.1
source activate SupCon

python main_supcon.py \
--batch_size 160 \
--model resnet50 \
--size 224 \
--learning_rate 0.1 \
--temp 0.1 \
--dataset path \
--data_folder ../lsz_kaggle_reorganized/train/image \
--method SimCLR  \
--mean "(0.3809, 0.2645, 0.1888)" \
--std "(0.2911, 0.2111, 0.1711)" \
--num_workers 10 \
--trial train_according2issue_png_1210