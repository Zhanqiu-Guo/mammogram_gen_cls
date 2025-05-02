#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128GB
#SBATCH --mail-user=zg2238@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --job-name branched-vae-seed42
#SBATCH --output mammogram_test_sbatch_output/branched_vae/resnet_vae_classifier_age.out

module load gcc/8.1.0
module load git

source /gpfs/home/zg2238/.bashrc 
conda activate ai4health
cd /gpfs/scratch/zg2238/mammogram_test

python -u resnet_vae_classifier_age.py



