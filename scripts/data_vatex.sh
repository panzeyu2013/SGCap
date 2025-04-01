#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate GCap

python ../dataset/data_prepare/vatex_prepare.py \
    --text_path "../data/origin/VATEX/annotation/train_val_test.json" \
    --video_path "../data/origin/VATEX/VATEX-videos" \
    --save_path "../data/extract/vatex" \
    --text_split "all"

python ../dataset/data_prepare/occurance.py \
    --data_path "../data/extract/vatex/all_captions.pth" \
    --output_path "../data/extract/vatex/all_nv_jaccard.pth" \
    --num_workers 24
    # num_works parameter controls the number of process used for extraction