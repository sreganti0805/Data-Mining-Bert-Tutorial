"""
predict.py — Run inference on a single essay or a CSV file.

Usage:
    # Score a single essay typed inline:
    python predict.py --text "My essay text here..."

    # Score all essays in a CSV and save results:
    python predict.py --csv essay.csv --output predictions.csv
"""

import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from config import *


def load_model(device):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device).eval()
    return tokenizer, model


def predict_score(text, tokenizer, model, device):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        output = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
        )
    return torch.argmax(output.logits, dim=1).item() + 1  # back to 1-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Single essay string to score')
    parser.add_argument('--csv',  type=str, help='CSV file with a full_text column')
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = load_model(device)

    if args.text:
        score = predict_score(args.text, tokenizer, model, device)
        print(f"Predicted score: {score}")

    elif args.csv:
        df = pd.read_csv(args.csv)
        df['predicted_score'] = df['full_text'].apply(
            lambda t: predict_score(str(t), tokenizer, model, device)
        )
        df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
