"""
Semi-Supervised Learning for Sentiment Analysis (self-training).

Uses a small labeled set + high-confidence pseudo-labels from unlabeled data.
Training is on combined data; process repeats for multiple iterations.

Usage:
    # Command line (recommended)
    python semi_supervised_training.py --labeled data/covidvaccine_roberta_sentiment.csv \\
        --unlabeled data/covidvaccine.csv --labeled-size 1000 --unlabeled-size 20000 \\
        --confidence 0.9 --iterations 2 --output-dir ./best_model

    # From Python
    from semi_supervised_training import semi_supervised_training
    model, tokenizer = semi_supervised_training(
        labeled_data_path="data/labeled.csv",
        unlabeled_data_path="data/unlabeled.csv",
        labeled_sample_size=1000,
        unlabeled_sample_size=20000,
        confidence_threshold=0.9,
        num_iterations=2,
        output_dir="./best_model",
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_BATCH_SIZE = 4  # Small default to avoid MPS OOM on limited GPU memory
MAX_LENGTH = 512
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence for pseudo-labeling

def get_device(use_cpu=False):
    """Get the best available device. Set use_cpu=True to avoid MPS/CUDA OOM."""
    if use_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SentimentDataset(Dataset):
    """Dataset for sentiment analysis"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_model(use_cpu=False):
    """Load pre-trained model. use_cpu=True forces CPU (avoids MPS OOM)."""
    print("Loading pre-trained model...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    device = get_device(use_cpu=use_cpu)
    model = model.to(device)

    print(f"Model loaded, using device: {device}")
    return model, tokenizer, config, device

def predict_with_confidence(model, tokenizer, texts, device, id2label, batch_size=DEFAULT_BATCH_SIZE):
    """Predict with confidence scores."""
    model.eval()
    predictions = []
    confidences = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_texts = [str(t).strip() or "" for t in (batch.tolist() if hasattr(batch, "tolist") else list(batch))]

        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probs, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]

        predictions.extend(predicted_ids.cpu().numpy())
        confidences.extend(max_probs.cpu().numpy())

        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    return np.array(predictions), np.array(confidences)

def pseudo_label_unlabeled_data(model, tokenizer, unlabeled_texts, device, id2label, confidence_threshold=0.9, batch_size=DEFAULT_BATCH_SIZE):
    """Generate pseudo-labels for unlabeled data."""
    print(f"\nGenerating pseudo-labels for {len(unlabeled_texts)} unlabeled texts...")
    print(f"Confidence threshold: {confidence_threshold}, batch_size: {batch_size}")

    predictions, confidences = predict_with_confidence(model, tokenizer, unlabeled_texts, device, id2label, batch_size=batch_size)
    
    # Filter high-confidence predictions
    high_confidence_mask = confidences >= confidence_threshold
    pseudo_labeled_texts = unlabeled_texts[high_confidence_mask]
    pseudo_labels = predictions[high_confidence_mask]
    pseudo_confidences = confidences[high_confidence_mask]
    
    print(f"High-confidence predictions: {len(pseudo_labeled_texts)} / {len(unlabeled_texts)}")
    pct = len(pseudo_labeled_texts) / len(unlabeled_texts) * 100 if len(unlabeled_texts) else 0
    print(f"Percentage: {pct:.2f}%")
    
    return pseudo_labeled_texts, pseudo_labels, pseudo_confidences

def train_model(model, train_dataset, val_dataset, tokenizer, device, num_epochs=3, output_dir="./best_model", batch_size=DEFAULT_BATCH_SIZE):
    """Train the model."""
    print(f"\nTraining model for {num_epochs} epochs (batch_size={batch_size})...")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            if device.type == "mps":
                torch.mps.synchronize()

        if device.type == "mps":
            torch.mps.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Saving best model...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
    
    return model

def     semi_supervised_training(
        labeled_data_path,
        unlabeled_data_path,
        labeled_sample_size=None,
        unlabeled_sample_size=None,
        confidence_threshold=0.9,
        num_iterations=1,
        output_dir="./best_model",
        batch_size=DEFAULT_BATCH_SIZE,
        use_cpu=False,
    ):
    """
    Semi-supervised training pipeline (self-training).

    Uses a small labeled set + high-confidence pseudo-labels from unlabeled data,
    then trains on combined data. Repeats for num_iterations.

    Args:
        labeled_data_path: CSV with 'text' and 'sentiment' columns.
        unlabeled_data_path: CSV with 'text' column.
        labeled_sample_size: Max labeled examples to use (None = all).
        unlabeled_sample_size: Max unlabeled examples for pseudo-labeling (None = all).
        confidence_threshold: Min confidence to accept a pseudo-label (e.g. 0.9).
        num_iterations: Self-training loops.
        output_dir: Where to save the best model.
        batch_size: Training and inference batch size (reduce if MPS OOM).
        use_cpu: Force CPU to avoid MPS/CUDA OOM.
    """
    print("=" * 60)
    print("Semi-Supervised Learning for Sentiment Analysis")
    print("=" * 60)
    
    # Load labeled data
    print(f"\nLoading labeled data from: {labeled_data_path}")
    labeled_df = pd.read_csv(labeled_data_path)
    
    if 'sentiment' not in labeled_df.columns:
        raise ValueError("Labeled data must have 'sentiment' column")
    
    # Sample labeled data if specified
    if labeled_sample_size and len(labeled_df) > labeled_sample_size:
        print(f"Sampling {labeled_sample_size} labeled examples...")
        labeled_df = labeled_df.sample(n=labeled_sample_size, random_state=42)
    
    # Map sentiment labels to IDs
    label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    labeled_df['label_id'] = labeled_df['sentiment'].map(label_to_id)
    labeled_df = labeled_df[labeled_df['label_id'].notna()].copy()
    
    print(f"Labeled data: {len(labeled_df)} examples")
    print(f"Label distribution:")
    print(labeled_df['sentiment'].value_counts())
    
    # Load unlabeled data
    print(f"\nLoading unlabeled data from: {unlabeled_data_path}")
    unlabeled_df = pd.read_csv(unlabeled_data_path)
    unlabeled_df = unlabeled_df[unlabeled_df['text'].notna() & (unlabeled_df['text'].str.strip() != '')].copy()
    if unlabeled_sample_size and len(unlabeled_df) > unlabeled_sample_size:
        print(f"Sampling {unlabeled_sample_size} unlabeled examples for pseudo-labeling...")
        unlabeled_df = unlabeled_df.sample(n=unlabeled_sample_size, random_state=42)
    print(f"Unlabeled data: {len(unlabeled_df)} examples")

    model, tokenizer, config, device = load_model(use_cpu=use_cpu)
    if use_cpu:
        print("Using CPU (use_cpu=True). Training will be slower but avoids GPU OOM.")
    
    # Split labeled data into train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        labeled_df['text'].tolist(),
        labeled_df['label_id'].astype(int).tolist(),
        test_size=0.2,
        random_state=42,
        stratify=labeled_df['label_id']
    )
    
    print(f"\nTrain set: {len(train_texts)} examples")
    print(f"Validation set: {len(val_labels)} examples")
    
    # Self-training iterations
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Self-Training Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Generate pseudo-labels
        unlabeled_texts = unlabeled_df['text'].tolist()
        pseudo_texts, pseudo_labels, pseudo_confidences = pseudo_label_unlabeled_data(
            model, tokenizer, np.array(unlabeled_texts), device, id_to_label,
            confidence_threshold=confidence_threshold, batch_size=batch_size,
        )
        
        # Combine labeled and pseudo-labeled data
        combined_texts = train_texts + pseudo_texts.tolist()
        combined_labels = train_labels + pseudo_labels.astype(int).tolist()
        
        print(f"\nCombined training set:")
        print(f"  Labeled: {len(train_texts)}")
        print(f"  Pseudo-labeled: {len(pseudo_texts)}")
        print(f"  Total: {len(combined_texts)}")
        
        # Create datasets
        train_dataset = SentimentDataset(combined_texts, combined_labels, tokenizer, MAX_LENGTH)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
        
        model = train_model(model, train_dataset, val_dataset, tokenizer, device, NUM_EPOCHS, output_dir, batch_size)
        
        # Update confidence threshold for next iteration (optional)
        if iteration < num_iterations - 1:
            confidence_threshold = min(0.95, confidence_threshold + 0.02)
            print(f"\nUpdating confidence threshold to {confidence_threshold} for next iteration")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best model saved to: {output_dir}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Semi-supervised sentiment training (self-training) with labeled + pseudo-labeled data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labeled", "-l", required=True, help="Path to labeled CSV (columns: text, sentiment)")
    parser.add_argument("--unlabeled", "-u", required=True, help="Path to unlabeled CSV (column: text)")
    parser.add_argument("--labeled-size", type=int, default=None, help="Max labeled examples to use (default: all)")
    parser.add_argument("--unlabeled-size", type=int, default=50000, help="Max unlabeled examples for pseudo-labeling")
    parser.add_argument("--confidence", "-c", type=float, default=0.9, help="Min confidence for pseudo-labels")
    parser.add_argument("--iterations", "-i", type=int, default=2, help="Self-training iterations")
    parser.add_argument("--output-dir", "-o", default="./best_model", help="Output directory for best model")
    parser.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (reduce if MPS OOM)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only (avoids MPS OOM)")
    args = parser.parse_args()

    if not os.path.exists(args.labeled):
        raise FileNotFoundError(f"Labeled file not found: {args.labeled}")
    if not os.path.exists(args.unlabeled):
        raise FileNotFoundError(f"Unlabeled file not found: {args.unlabeled}")

    semi_supervised_training(
        labeled_data_path=args.labeled,
        unlabeled_data_path=args.unlabeled,
        labeled_sample_size=args.labeled_size,
        unlabeled_sample_size=args.unlabeled_size,
        confidence_threshold=args.confidence,
        num_iterations=args.iterations,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        use_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()

