#!/bin/bash --login

#SBATCH --job-name=RelClass

#SBATCH --output=logs/out_relation_classifier.txt
#SBATCH --error=logs/err_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=16G

#SBATCH -t 0-01:00:00

#SBATCH --gres=gpu:1

conda activate venv

python3 importance_classifier_production.py

echo 'Job Finished !!!'