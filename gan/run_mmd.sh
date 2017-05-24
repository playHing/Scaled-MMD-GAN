#!/usr/bin/env bash

if [[ $1 == 'sample' ]]; then
    is_train=False
    echo "Sampling"
else
    is_train=True
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_mmd.py \
    --model=mmd \
    --use_kernel --is_train=True \
    --name=mmd \
    --dataset=lsun \
    --max_iteration=5000 --init=0.1 --learning_rate=.00011 --batch_size=128 \
    --dataset=mnist --architecture=dc --dc_discriminator --kernel=rbf \
