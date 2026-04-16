# config.py — Edit these settings before running

DATA_PATH      = 'essay.csv'         # path to your CSV file
MODEL_NAME = '/project/sreganti/bert_model'
MODEL_SAVE_PATH = 'best_bert_essay.pt'

MAX_LEN    = 128   # max token length (reduce to 256 if you hit memory issues)
BATCH_SIZE = 2     # reduce to 4 on low-VRAM GPUs
EPOCHS     = 1
LR         = 2e-5
NUM_LABELS = 6     # scores 1-6
SEED       = 42