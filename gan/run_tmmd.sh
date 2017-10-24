#!/usr/bin/env bash

if [[ $1 == 'sample' ]]; then
    is_train=False
    echo "Sampling"
else
    is_train=True
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_mmd.py \
    --model=tmmd \
    --suffix= \
    --use_kernel --is_train=True \
    --threads=2 \
    --name=temp \
    --max_iteration=5000 --init=0.1 \
    --learning_rate=.000140 --batch_size=128 \
    --architecture=dfc --dc_discriminator \
    --kernel=mix_rq \
    --dsteps=5 --start_dsteps=5 \
    --batch_norm \
    --dataset=cifar10 \
    --gradient_penalty=10.0 \
    --log=False
