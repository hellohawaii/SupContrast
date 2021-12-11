#!/bin/bash

# 加载环境
module load anaconda/2020.11
module load cuda/11.1
source activate torch

python main_supcon.py \
--batch_size 160 \
--model resnet50 \
--learning_rate 0.1 \
--temp 0.1 \
--dataset path \
--data_folder ../masked_region \
--method SimCLR  \
--num_workers 10 \
--trial train_png_masked_region_1211