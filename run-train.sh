#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

RUNDIR=runs-dir
rm -rf $RUNDIR/*

# # old dataset
# python train.py \
#     --train_evaluator 0 \
#     --dataset_root_folder unified_grasp_data/ \
#     --logdir $RUNDIR/ \
#     --gan 0 \
#     --allowed_categories single_object

# acronym
python train.py \
    --train_evaluator 0 \
    --dataset_root_folder /data-dir/ \
    --logdir $RUNDIR/ \
    --gan 0 \
    --num_objects_per_batch 1 \
    --num_grasps_per_object 10 \
    --train_split_fp dataset-acr-train.yml \
    --acronym --full_pc
