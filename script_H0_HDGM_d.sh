#!/bin/bash
# This is our first script.
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 25
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 20
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 15
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 10
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 5
CUDA_VISIBLE_DEVICES=3 python main_HDGM_H0.py --d 3
