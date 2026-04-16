# BERT Essay Scorer – Data Mining Tutorial

A step-by-step tutorial for fine-tuning a BERT model to automatically score student essays (scores **1–6**) using the University of Memphis **iTiger GPU cluster**.

This project demonstrates how to train a deep learning model for **Automated Essay Scoring (AES)** using **PyTorch**, **Hugging Face Transformers**, and **SLURM** on an HPC environment.

---

# Overview

This tutorial walks you through fine-tuning a BERT model on **3,000 real student essays**. By the end, you will have a trained essay scoring model running on the iTiger cluster through VS Code Remote SSH.

### Approximate Results

Typical outcomes may vary depending on random seed, hardware, and environment.

- ~0.45 QWK after 1 epoch  
- ~0.52–0.56 QWK after 2 epochs  
- ~0.58–0.62 QWK after 3 epochs  

A QWK score around **0.60** represents substantial agreement with human graders.

---

# What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is a pre-trained language model developed by Google. It has already learned the structure and meaning of English from billions of words.

Instead of training from scratch, we **fine-tune** BERT for essay scoring by adding a classification head that predicts scores from **1 to 6**.

---

# What is QWK?

**Quadratic Weighted Kappa (QWK)** is the standard evaluation metric for essay scoring.

Unlike plain accuracy, QWK penalizes predictions that are far from the true score more heavily than nearby ones.

Example:
- Predicting **5 instead of 6** = small penalty  
- Predicting **1 instead of 6** = large penalty

---

# Project Structure

```text
Data-Mining-Bert-Tutorial/
├── config.py
├── train.py
├── predict.py
├── run_train.sh
├── requirements.txt
├── essay.csv
├── README.md
└── DataMining_BERT_Tutorial.docx
