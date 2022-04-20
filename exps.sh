#!/bin/bash

opts="adam adagrad nesterov_sgd"
models="mlp[16384,16384,512] preresnet18"
regs="wd dropout none"
imbalance="yes no"
cl="log exp step linear"

#for model in $models; do
#  for opt in $opts; do
#    for reg in $regs; do
#        echo nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt $opt --arch \'$model\' --reg $reg --iid \> logs/log.txt \&
#        echo nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt $opt --arch \'$model\' --reg $reg \> logs/log.txt \&
#      done
#    done
#  done

for c in $cl; do
        echo nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt nesterov_sgd --arch preresnet18 --reg none --curriculum $c \> logs/log.txt \&

done

#model:preresnet18|reg:wd|iid:False|opt:adam|cls_imb:-1
#nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adam --arch 'preresnet18' --reg wd > logs/log.txt &
#
#model:preresnet18|reg:wd|iid:True|opt:adam|cls_imb:-1
#nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adam --arch 'preresnet18' --reg wd --iid > logs/log.txt &
#
#model:preresnet18|reg:none|iid:True|opt:adam|cls_imb:0
#nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adam --arch 'preresnet18' --reg none --iid --cls_imb 0 > logs/log.txt &
#
#model:preresnet18|reg:none|iid:False|opt:adam|cls_imb:-1
#nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adam --arch 'preresnet18' --reg none > logs/log.txt &
#
#model:preresnet18|reg:none|iid:True|opt:adagrad|cls_imb:-1
#nohup python train.py --batchsize 1028 --workers 2 --epochs 80 --opt adagrad --arch 'preresnet18' --reg none > logs/log.txt &
