#!/bin/bash
# NOTE: Replace all instances of 'sreganti' below with your iTiger username
#SBATCH --job-name=bert_essay
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_5000:1
#SBATCH --time=02:00:00
#SBATCH --output=/project/sreganti/bert_essay/bert_essay/slurm-%j.out                       
#SBATCH --error=/project/sreganti/bert_essay/bert_essay/slurm-%j.err                        

cd /project/sreganti/bert_essay/bert_essay
/project/sreganti/miniconda3/bin/python -u train.py
