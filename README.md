# BERT Essay Scorer – Data Mining Tutorial

A step-by-step tutorial for fine-tuning a BERT model to automatically score student essays (scores 1–6) using the University of Memphis iTiger GPU cluster.

## Contents
- `train.py` – Fine-tunes bert-base-uncased on the essay dataset
- `predict.py` – Runs inference on a single essay or a CSV file
- `config.py` – All hyperparameters and paths in one place
- `run_train.sh` – SLURM job script for iTiger (bigTiger partition, RTX 5000)
- `requirements.txt` – Python dependencies
- `essay.csv` – ~3,000 student essays with scores 1–6
- `DataMining_BERT_Tutorial.docx` – Full tutorial walkthrough

## Expected Results
~0.60 Quadratic Weighted Kappa (QWK) after 3 epochs — substantial agreement with human graders.

## Requirements
- Access to the University of Memphis iTiger HPC cluster
- VS Code with Remote – SSH extension
- Python 3.8+, PyTorch 2.5.1 (cu121), HuggingFace Transformers

## Usage
Follow the steps in `DataMining_BERT_Tutorial.docx`. The short version:
```bash
git clone https://github.com/sreganti0805/Data-Mining-Bert-Tutorial.git
cd Data-Mining-Bert-Tutorial
sbatch run_train.sh
```
