import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm
import time

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32  # Batch size, adjust based on GPU memory
MAX_LENGTH = 512  # Max text length

def get_device():
    """Get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model():
    """Load model and configuration"""
    print("Loading model...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    
    device = get_device()
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded, using device: {device}")
    return model, tokenizer, config, device

def predict_batch(model, tokenizer, texts, device, id2label):
    """Perform sentiment analysis on a batch of texts"""
    # Tokenize
    tokens = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    
    # Model inference
    with torch.no_grad():
        output = model(**tokens)
    
    # Get predictions
    logits = output.logits
    probs = F.softmax(logits, dim=-1)
    predicted_class_ids = torch.argmax(probs, dim=-1)
    
    # Convert to list
    predictions = []
    for i in range(len(texts)):
        class_id = predicted_class_ids[i].item()
        label = id2label[class_id]
        confidence = probs[i][class_id].item()
        
        predictions.append({
            'sentiment': label,
            'sentiment_id': class_id,
            'confidence': confidence,
            'negative_prob': probs[i][0].item(),
            'neutral_prob': probs[i][1].item(),
            'positive_prob': probs[i][2].item()
        })
    
    # Synchronize MPS if needed
    if device.type == 'mps':
        torch.mps.synchronize()
    
    return predictions

def analyze_csv_file(csv_path, output_path=None, sample_size=None):
    """Perform sentiment analysis on entire CSV file"""
    print(f"\nLoading data file: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Total rows: {len(df)}")
    
    # Sample data if sample_size specified
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} rows for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Check for text column
    if 'text' not in df.columns:
        raise ValueError("'text' column not found in CSV file")
    
    # Remove empty texts
    df = df[df['text'].notna() & (df['text'].str.strip() != '')].copy()
    print(f"Valid text rows: {len(df)}")
    
    # Load model
    model, tokenizer, config, device = load_model()
    
    # Get label mapping
    if hasattr(config, 'id2label'):
        id2label = config.id2label
    else:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    
    print(f"Label mapping: {id2label}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"\nStarting sentiment analysis...")
    
    # Batch processing
    all_predictions = []
    texts = df['text'].tolist()
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing batches", total=total_batches):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_predictions = predict_batch(model, tokenizer, batch_texts, device, id2label)
        all_predictions.extend(batch_predictions)
    
    elapsed_time = time.time() - start_time
    
    # Add predictions to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    df_result = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Sentiment analysis complete!")
    print("=" * 60)
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {len(texts)/elapsed_time:.2f} texts/sec")
    print(f"\nSentiment distribution:")
    print(df_result['sentiment'].value_counts())
    print(f"\nSentiment distribution percentage:")
    print(df_result['sentiment'].value_counts(normalize=True) * 100)
    
    # Save results
    if output_path is None:
        output_path = csv_path.replace('.csv', '_roberta_sentiment.csv')
    
    df_result.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nResults saved to: {output_path}")
    
    return df_result

if __name__ == "__main__":
    csv_file = "data/covidvaccine.csv"
    
    # Test with small sample first
    analyze_csv_file(csv_file, sample_size=1000)
    
    # Process entire file (or use sample_size for testing)
    print("=" * 60)
    print("Twitter RoBERTa Sentiment Analysis - Batch Processing Mode")
    print("=" * 60)
    
    # Process entire file (use sample_size if data is too large)
    # result_df = analyze_csv_file(
    #     csv_file, 
    #     output_path="data/covidvaccine_roberta_sentiment.csv",
    #     sample_size=None  # Set to None for full data, or number for testing
    # )
    
    print("\nAnalysis complete!")