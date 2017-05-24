#!/usr/bin/env bash

if [[ $1 == 'sample' ]]; then
    is_train=False
    echo "Sampling"
else
    is_train=True
fi

python main_mmd.py \
    --model=tmmd \
    --use_kernel --is_train=True \
    --name=tmmd\
    --max_iteration=20000 --init=0.1 --learning_rate=.00098 --batch_size=128 \
    --dataset=cifar10 --architecture=dc --dc_discriminator --kernel=rq \
##    --gradient_penalty=10.0

##python main_mmd.py \
##    --model=tmmd \
##    --use_kernel --is_train=True \
##    --name=tmmd\
##    --max_iteration=20000 --init=0.1 --learning_rate=.00095 --batch_size=512 \
##    --dataset=cifar10 --architecture=mlp --dc_discriminator --kernel=rq \
