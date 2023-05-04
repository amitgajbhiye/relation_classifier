#!/bin/bash --login

#SBATCH --job-name=W2vRelClass

#SBATCH --output=logs/out_fasttext_data_relation_classifier.txt
#SBATCH --error=logs/err_fasttext_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=14G

#SBATCH -t 0-10:00:00

#SBATCH --gres=gpu:1

conda activate relbert

python3 importance_classifier_production.py

echo 'Job Finished !!!'