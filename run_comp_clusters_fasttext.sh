#!/bin/bash --login

#SBATCH --job-name=FastTRelClass

#SBATCH --output=logs/out_complementary_clusters_fasttext_data_relation_classifier.txt
#SBATCH --error=logs/err__complementary_clusters_fasttext_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=20G

#SBATCH -t 0-12:00:00

#SBATCH --gres=gpu:2

source venv_relbert/bin/activate

module load CUDA/11.7

python3 importance_classifier_production.py "/scratch/c.scmag3/commonality_detection/output_files/complementary_clusters_files/fasttext_complementary_clusters.txt" "/scratch/c.scmag3/commonality_detection/output_files/complementary_clusters_files/fasttext_complementary_clusters_relbert_scores.txt"

echo 'Job Finished !!!'
