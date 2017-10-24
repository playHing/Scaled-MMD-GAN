#!/usr/bin/env bash

if [[ $1 == 'sample' ]]; then
    is_train=False
    echo "Sampling"
else
    is_train=True
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python main_mmd.py \
    --model=mmd \
    --suffix= \
    --use_kernel --is_train=True \
    --threads=2 \
    --name=mmd \
    --max_iteration=50000 --init=0.1 \
    --learning_rate=.000104 --batch_size=128 --real_batch_size=128 \
    --architecture=dcgan --dc_discriminator \
    --kernel=distance \
    --dsteps=5 --start_dsteps=5 \
    --batch_norm \
    --dataset=mnist \
    --gradient_penalty=10.0 --output_size=64 \
    --gf_dim=32 --df_dim=32 --dof_dim=16 \
    --log=False
