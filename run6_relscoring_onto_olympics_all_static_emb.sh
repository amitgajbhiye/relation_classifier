#!/bin/bash --login

#SBATCH --job-name=olympics

#SBATCH --output=logs/ontology_completion/out_relscoring_onto_olympics_vocab_all_static_emb.txt
#SBATCH --error=logs/ontology_completion/err_relscoring_onto_olympics_vocab_all_static_emb.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=20G

#SBATCH -t 0-10:00:00

#SBATCH --gres=gpu:2

source venv_relbert/bin/activate

module load CUDA/11.7

python3 importance_classifier_production.py /scratch/c.scmag3/commonality_detection/output_files/ontology_completion_similar_thresh_50/olympics

echo 'Job Finished !!!'
