import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, cohen_kappa_score
from config import *

class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'label': torch.tensor(self.labels[idx], dtype=torch.long)}

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['label'].to(device))
        total_loss += outputs.loss.item()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), labels=batch['label'].to(device))
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    return total_loss/len(loader), np.mean(np.array(all_preds)==np.array(all_labels)), cohen_kappa_score(all_labels, all_preds, weights='quadratic'), all_preds, all_labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['score'] - 1
    df['text'] = df['full_text'].str.strip()
    print(f"Loaded {len(df)} essays.", flush=True)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=SEED)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}", flush=True)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(EssayDataset(train_df['text'], train_df['label'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EssayDataset(val_df['text'], val_df['label'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    test_loader = DataLoader(EssayDataset(test_df['text'], test_df['label'], tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    print("Loading model...", flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    best_val_qwk = -1
    for epoch in range(1, EPOCHS + 1):
        print(f"Starting epoch {epoch}/{EPOCHS}...", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_acc, val_qwk, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val QWK: {val_qwk:.4f}", flush=True)
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  --> Best model saved.", flush=True)
    print("\n--- Test Results ---", flush=True)
    
    test_loss, test_acc, test_qwk, test_preds, test_labels = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test QWK: {test_qwk:.4f}", flush=True)
    print(classification_report([l+1 for l in test_labels], [p+1 for p in test_preds], target_names=[f"Score {i}" for i in range(1,7)]), flush=True)

if __name__ == '__main__':
    main()