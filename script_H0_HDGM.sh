#!/bin/bash
# This is our first script.
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 3000 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 2500 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 2000 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 1500 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 1000 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 500 --d 30
CUDA_VISIBLE_DEVICES=2 python main_HDGM_H0.py --n 100 --d 30
