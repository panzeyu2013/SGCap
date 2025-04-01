#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate GCap

python ../dataset/data_prepare/msvd_prepare.py \
    --text_path "../data/origin/msvd/annotation/all_caption.json" \
    --video_path "../data/origin/msvd/video" \
    --save_path "../data/extract/msvd" \
    --text_split "all"

python ../dataset/data_prepare/occurance.py \
    --data_path "../data/extract/msvd/all_captions.pth" \
    --output_path "../data/extract/msvd/all_nv_jaccard.pth" \
    --num_workers 24
    # num_works parameter controls the number of process used for extraction