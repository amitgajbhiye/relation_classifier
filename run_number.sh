#!/bin/bash --login

#SBATCH --job-name=numBRelClass

#SBATCH --output=logs/out_numberbatch_data_relation_classifier.txt
#SBATCH --error=logs/err_numberbatch_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=32G
#SBATCH --exclusive

#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

conda activate relbert

python3 importance_classifier_production.py "datasets/rel_inp_numberbatch_ueft_label_similar_0.5thresh_count_20thresh.txt" "output_files/numberbatch_relation_probs.txt"

echo 'Job Finished !!!'