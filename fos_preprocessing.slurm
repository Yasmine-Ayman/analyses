#!/bin/bash
#SBATCH -t 0-4:00
#SBATCH -p short
#SBATCH -J fos_preprocessing
#SBATCH -o fos_preprocessing_%j.log
#SBATCH -e fos_preprocessing_%j.log
#SBATCH -c 1
#SBATCH --mem=100GB

mouse=$1
date=$2
session=$3
options=$4

echo "Running FOS preprocessing for ${mouse} ${date} ${session}."

module load conda/miniforge3
conda activate mouse_imaging_rhel

python $CODE/analyses/fos_preprocessing.py --mouse $mouse --date $date --session $session --ops $options
