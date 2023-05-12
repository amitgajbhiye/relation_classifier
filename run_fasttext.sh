#!/bin/bash --login

#SBATCH --job-name=FastTRelClass

#SBATCH --output=logs/out_cuda_fasttext_data_relation_classifier.txt
#SBATCH --error=logs/err_cuda_fasttext_data_relation_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=15G

#SBATCH -t 0-03:00:00

#SBATCH --gres=gpu:1

source venv_relbert/bin/activate

module load CUDA/11.7

python3 importance_classifier_production.py "datasets/rel_inp_fasttext_ueft_label_similar_0.5thresh_count_100thresh.txt" "output_files/fasttext_relation_probs.txt"

echo 'Job Finished !!!'