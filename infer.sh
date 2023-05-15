#!/usr/bin/bash

python3 inference.py results/model_epoch_399.pth 1 jetson time
python3 inference.py results/model_epoch_399ema.pth 1 jetson time
python3 inference.py results/model_epoch_399.pth 1 jetson mem
python3 inference.py results/model_epoch_399ema.pth 1 jetson mem
python3 inference.py results/model_epoch_399.pth 2 jetson time
python3 inference.py results/model_epoch_399ema.pth 2 jetson time
python3 inference.py results/model_epoch_399.pth 2 jetson mem
python3 inference.py results/model_epoch_399ema.pth 2 jetson mem
python3 inference.py results/model_epoch_399.pth 5 jetson time
python3 inference.py results/model_epoch_399ema.pth 5 jetson time
python3 inference.py results/model_epoch_399.pth 5 jetson mem
python3 inference.py results/model_epoch_399ema.pth 5 jetson mem