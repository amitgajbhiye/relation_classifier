#!/bin/bash --login

#SBATCH --job-name=W2vRelClass

#SBATCH --output=logs/out_word2vec_data_relation_classifier.txt
#SBATCH --error=logs/err_word2vec_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10G

#SBATCH -t 0-00:30:00

#SBATCH --gres=gpu:1

conda activate relbert

python3 importance_classifier_production.py "datasets/rel_inp_word2vec_ueft_label_similar_0.5thresh_count_10thresh.txt" "output_files/w2v_relation_probs.txt"

echo 'Job Finished !!!'