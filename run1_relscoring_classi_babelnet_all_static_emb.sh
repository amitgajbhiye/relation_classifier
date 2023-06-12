#!/bin/bash --login

#SBATCH --job-name=FastTRelClass

#SBATCH --output=logs/bablenet_domain/out_relscoring_classi_babelnet_domain_all_static_emb.txt
#SBATCH --error=logs/bablenet_domain/out_relscoring_classi_babelnet_domain_all_static_emb.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=20G

#SBATCH -t 0-12:00:00

#SBATCH --gres=gpu:2

source venv_relbert/bin/activate

module load CUDA/11.7

python3 importance_classifier_production.py /scratch/c.scmag3/commonality_detection/output_files/classification_vocabs_similar_thresh_50/bablenet_domain

echo 'Job Finished !!!'
