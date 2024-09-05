#!/bin/bash

echo "Starting"

NGPUS=2

cd experiments/geotransformer.rellis.filtered
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=$NGPUS trainval.py
