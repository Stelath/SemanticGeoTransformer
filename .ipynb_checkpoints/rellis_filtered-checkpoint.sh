#!/bin/bash

echo "Starting"

cd experiments/geotransformer.rellis.filtered
CUDA_VISIBLE_DEVICES=0 python trainval.py
