#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate GCap

python ../dataset/data_prepare/msrvtt_prepare.py \
    --text_path "../data/origin/msrvtt/train_val_test_annotation/train_val_test_videodatainfo.json" \
    --video_path "../data/origin/msrvtt/train_val_test_video" \
    --save_path "../data/extract/msrvtt" \
    --text_split "all"

python ../dataset/data_prepare/occurance.py \
    --data_path "../data/extract/msrvtt/all_captions.pth" \
    --output_path "../data/extract/msrvtt/all_nv_jaccard.pth" \
    --num_workers 24
    # num_works parameter controls the number of process used for extraction