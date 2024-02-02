#!/bin/bash

echo "Starting"

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

micromamba activate geotransformer

cd experiments/geotransformer.rellis
CUDA_VISIBLE_DEVICES=0 python trainval.py