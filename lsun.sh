#!/usr/bin/env bash

MODEL=MMD
SAVEPATH=./
DATAPATH=./data
export TF_MIN_GPU_MULTIPROCESSOR_COUNT=3;
#export CUDA_VISIBLE_DEVICES=0;


if [[${MODEL} == 'MMD']] || [[${MODEL} == 'ALL']]; then
    python ./gan/main.py \
        --checkpoint_dir=${SAVEPATH}checkpoint \
        --sample_dir=${SAVEPATH}samples \
        --log_dir=${SAVEPATH}logs \
        --data_dir={DATAPATH} \
        --model=mmd --name=mmd --kernel=mix_rq_1dot \
        --architecture=dcgan --output_size=64 --dof_dim=16 \
        --gradient_penalty=1. --L2_discriminator_penalty=1. \
        --dataset=lsun \
        --MMD_lr_scheduler \
fi

if [[${MODEL} == 'WGAN-GP']] || [[${MODEL} == 'ALL']]; then
    python ./gan/main.py \
        --checkpoint_dir=${SAVEPATH}checkpoint \
        --sample_dir=${SAVEPATH}samples \
        --log_dir=${SAVEPATH}logs \
        --data_dir={DATAPATH} \
        --model=wgan_gp --name=wgan_gp \
        --architecture=dcgan --output_size=64 --dof_dim=1 \
        --gradient_penalty=10. \
        --dataset=lsun \
        --MMD_lr_scheduler \
fi

if [[${MODEL} == 'CRAMER']] || [[${MODEL} == 'ALL']]; then
    python ./gan/main.py \
        --checkpoint_dir=${SAVEPATH}checkpoint \
        --sample_dir=${SAVEPATH}samples \
        --log_dir=${SAVEPATH}logs \
        --data_dir={DATAPATH} \
        --model=cramer --name=cramer_gan \
        --architecture=dcgan --output_size=64 --dof_dim=256 \
        --gradient_penalty=10. \
        --dataset=lsun \
        --MMD_lr_scheduler \
fi


