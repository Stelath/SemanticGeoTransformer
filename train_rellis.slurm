#!/bin/bash

#SBATCH --job-name rellis-trian 
#SBATCH --mem 250gb
#SBATCH --time 10:00:00
#SBATCH --gpus a100:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 64
#SBATCH -p viprgs



cd /project/bli4/maps/wacv/GeoTransformer/

APPTAINER_PATH="/project/niyanil/ailab/Tools/bin:$PATH"
HF_HOME="/project/nianyil/ailab/Tools/huggingface"

# Run Apptainer with ailab & micromamba mounted + Nvidia Support
apptainer run --nv --env "PATH=$APPTAINER_PATH" --env "HF_HOME=$HF_HOME" -B /project/nianyil/ailab:/project/nianyil/ailab -B /project/nianyil/ailab/Tools/micromamba:$HOME/micromamba -B $TMPDIR:$TMPDIR -B /scratch/$USER:/scratch/$USER /project/nianyil/ailab/Tools/ubuntu.sif bash /project/bli4/maps/wacv/GeoTransformer/train_rellis.sh
