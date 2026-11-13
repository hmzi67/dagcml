"""Self-contained isolation script for the mDeBERTa collapse-to-ln(3) bug.

Fine-tunes a backbone on a small English XNLI slice with a completely
standard loop (AdamW lr=2e-5 eps=1e-6, 10% linear warmup, clip 1.0) and
reports whether loss drops below ln(3) and whether val predictions spread
across all 3 classes. Model name and from_pretrained kwargs are set as
variables at the top so different configs (A/B/C/D from the debugging
protocol) can be tried by editing them and rerunning.

Usage:
    python debug_backbone.py
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset

# ============================================================================
# CONFIG — edit these to A/B test
# ============================================================================
MODEL_NAME = "microsoft/mdeberta-v3-base"
FROM_PRETRAINED_KWARGS = dict(dtype=torch.float32)   # e.g. add attn_implementation="eager"

SEED = 42
TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
BATCH_SIZE = 32
MAX_LENGTH = 128
LR = 2e-5
EPS = 1e-6
WARMUP_RATIO = 0.10
CLIP_NORM = 1.0
TOTAL_STEPS = 300
LOG_EVERY = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XNLIDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length):
        self.premises = hf_split["premise"]
        self.hypotheses = hf_split["hypothesis"]
        self.labels = hf_split["label"]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main():
    print(f"=== Config: MODEL_NAME={MODEL_NAME} kwargs={FROM_PRETRAINED_KWARGS} ===")

    print("Loading XNLI (en)...")
    train_raw = load_dataset("facebook/xnli", "en", split=f"train[:{TRAIN_SAMPLES}]")
    val_raw = load_dataset("facebook/xnli", "en", split=f"validation[:{VAL_SAMPLES}]")

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, **FROM_PRETRAINED_KWARGS
    )
    p = next(model.parameters())
    print(f"PARAM DTYPE: {p.dtype}")
    print(f"config torch_dtype: {getattr(model.config, 'torch_dtype', None)}")
    print(f"attn implementation: {getattr(model.config, '_attn_implementation', None)}")
    model.to(device)

    train_ds = XNLIDataset(train_raw, tokenizer, MAX_LENGTH)
    val_ds = XNLIDataset(val_raw, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR, eps=EPS)
    num_warmup = int(TOTAL_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup, num_training_steps=TOTAL_STEPS)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    step = 0
    running_loss = 0.0
    train_iter = iter(train_loader)
    final_loss = None

    while step < TOTAL_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1
        final_loss = loss.item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(f"step {step:4d}/{TOTAL_STEPS}  loss={avg:.4f}  (ln3={math.log(3):.4f})")
            running_loss = 0.0

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    dist = [int((all_preds == c).sum()) for c in range(3)]

    print("\n=== RESULTS ===")
    print(f"Config: {MODEL_NAME} kwargs={FROM_PRETRAINED_KWARGS}")
    print(f"Final (last-step) train loss: {final_loss:.4f}")
    print(f"Val accuracy: {acc:.4f}")
    print(f"Val prediction distribution (class 0/1/2): {dist}")
    if len(set(dist)) == 1 or max(dist) == len(all_preds):
        print("!!! COLLAPSE DETECTED: model predicts a single constant class !!!")
    else:
        print("OK: predictions spread across multiple classes")


if __name__ == "__main__":
    main()
