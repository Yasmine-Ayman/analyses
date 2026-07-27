#!/bin/bash
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH -J add_spatial_peaks_h
#SBATCH -o add_spatial_peaks_h_%j.log
#SBATCH -e add_spatial_peaks_h_%j.log
#SBATCH -c 1
#SBATCH --mem=100GB

mouse=$1
date=$2
session=$3
options=$4
layer=${5:-dF_sig}
n_shuffles=${6:-1000}

echo "Computing significant spatial peaks (_h, heading-filtered) for ${mouse} ${date} ${session}, layer=${layer}, n_shuffles=${n_shuffles}."

module load conda/miniforge3
conda activate mouse_imaging_rhel

python $CODE/analyses/add_spatial_peaks_h.py --mouse $mouse --date $date --session $session --ops $options --layer $layer --n_shuffles $n_shuffles
