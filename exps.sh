#!/bin/bash

opts="adam adagrad nesterov_sgd"
models="mlp[16384,16384,512] preresnet18"
regs="wd dropout none"
imbalance="yes no"

for model in $models; do
  for opt in $opts; do
    for reg in $regs; do
        echo nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt $opt --arch \'$model\' --reg $reg --iid \> logs/log.txt \&
        echo nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt $opt --arch \'$model\' --reg $reg \> logs/log.txt \&
      done
    done
  done

