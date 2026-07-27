#!/bin/bash
#SBATCH -t 0-6:00
#SBATCH -p short
#SBATCH -J add_dF_sig
#SBATCH -o add_dF_sig_%j.log
#SBATCH -e add_dF_sig_%j.log
#SBATCH -c 1
#SBATCH --mem=100GB

mouse=$1
date=$2
session=$3
options=$4

echo "Computing significant dF for ${mouse} ${date} ${session}."

module load conda/miniforge3
conda activate mouse_imaging_rhel

python $CODE/analyses/add_dF_sig.py --mouse $mouse --date $date --session $session --ops $options
