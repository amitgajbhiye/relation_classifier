#!/bin/bash --login

#SBATCH --job-name=W2vRelClass

#SBATCH --output=logs/out_complementary_clusters_word2vec_data_relation_classifier.txt
#SBATCH --error=logs/err_complementary_clusters_word2vec_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=12G

#SBATCH -t 0-2:00:00

#SBATCH --gres=gpu:2

conda activate relbert

python3 importance_classifier_production.py "/scratch/c.scmag3/commonality_detection/output_files/complementary_clusters_files/word2vec_complementary_clusters.txt" "/scratch/c.scmag3/commonality_detection/output_files/complementary_clusters_files/word2vec_complementary_clusters_relbert_scores.txt"

echo 'Job Finished !!!'