#!/bin/bash
#SBATCH -t 1-0:00
#SBATCH -p medium
#SBATCH -J preprocess_fos
#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -o preprocess_fos_%j.log
#SBATCH -e preprocess_fos_%j.log

mouse=$1
date=$2
session=$3
options=$4

source /n/app/conda2/etc/profile.d/conda.sh

nfiles=$TIER1DATA
scratch=$SCRATCH
data2=$O2DATA
session_branch=$mouse/$date/$session
session_ref=$session_branch/ref
raw2p=$scratch/2P/$session_branch
meanRef_dir=$data2/$session_branch/meanRef

echo "Processing ${mouse} ${date} ${session} with R1 pipeline."

# Transfer files to scratch
sbatch --wait $CODE/preprocess_2p/rsync_ssh.slurm 2P/$session_branch $nfiles $scratch

echo "Running FOS preprocessing for ${mouse} ${date} ${session}."

module load conda/miniforge3
conda activate mouse_imaging_rhel

python $CODE/analyses/fos_preprocessing.py --mouse $mouse --date $date --session $session --ops $options