#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py --mode 0 --batch_size 32 --workers 12 --nepoch 300 --model_name ecg --num_points 2048 --log_env ecg_2048_CD --lr 0.0001 --loss CD --use_mean_feature 0