#!/bin/bash --login

#SBATCH --job-name=numBRelClass

#SBATCH --output=logs/out_cuda_numberbatch_data_relation_classifier.txt
#SBATCH --error=logs/err_cuda_numberbatch_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10G
#SBATCH --exclusive

#SBATCH -t 0-00:30:00
#SBATCH --gres=gpu:1

module load CUDA/11.7


conda activate relbert

python3 importance_classifier_production.py "datasets/rel_inp_numberbatch_ueft_label_similar_0.5thresh_count_20thresh.txt" "output_files/numberbatch_relation_probs.txt"

echo 'Job Finished !!!'