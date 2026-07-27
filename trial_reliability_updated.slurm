#!/bin/bash
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH -J trial_reliability
#SBATCH -o trial_reliability_%j.log
#SBATCH -e trial_reliability_%j.log
#SBATCH -c 1
#SBATCH --mem=100GB

mouse=$1
date=$2
session=$3
options=$4
layer=${5:-dF_sig}
selective_only=${6:-False}
preferred_condition=${7:-peak_only}
y_bin_size=${8:-5}

echo "Computing trial reliability for ${mouse} ${date} ${session}, layer=${layer}."

module load conda/miniforge3
conda activate mouse_imaging_rhel

python $CODE/analyses/trial_reliability_updated.py \
    --mouse $mouse \
    --date $date \
    --session $session \
    --ops $options \
    --layer $layer \
    --selective_only $selective_only \
    --preferred_condition $preferred_condition \
    --y_bin_size $y_bin_size
