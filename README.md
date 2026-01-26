# COVID-19 Vaccine Twitter Sentiment Analysis

A comprehensive sentiment analysis pipeline for analyzing Twitter data related to COVID-19 vaccines using pre-trained RoBERTa models and batch processing.

## Features

- **Pre-trained Twitter RoBERTa Model**: Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` for accurate sentiment analysis
- **Batch Processing**: Efficient batch processing for large datasets
- **MPS/GPU Support**: Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- **Comprehensive Output**: Includes sentiment labels, confidence scores, and probability distributions

## Requirements

- Python 3.7+
- PyTorch 2.10.0+
- Transformers 4.57.6+
- pandas, numpy, tqdm

See `requirements.txt` for complete dependencies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/infiniteMJ/covid-big-data-batch-pipeline.git
cd covid-big-data-batch-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from pretrained_twitter_model import analyze_csv_file

# Analyze a CSV file with sentiment analysis
result_df = analyze_csv_file(
    csv_path="data/covidvaccine.csv",
    output_path="data/covidvaccine_roberta_sentiment.csv",
    sample_size=1000  # Set to None for full dataset
)
```

### Command Line

```bash
python pretrained_twitter_model.py
```

The script will:
1. Load the CSV file
2. Sample data (if `sample_size` is specified)
3. Load the pre-trained RoBERTa model
4. Perform batch sentiment analysis
5. Save results to a new CSV file

### Output Format

The output CSV includes all original columns plus:
- `sentiment`: Predicted sentiment (negative/neutral/positive)
- `sentiment_id`: Sentiment class ID (0/1/2)
- `confidence`: Prediction confidence score
- `negative_prob`: Probability of negative sentiment
- `neutral_prob`: Probability of neutral sentiment
- `positive_prob`: Probability of positive sentiment

## Model Details

- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Task**: Sentiment Classification
- **Classes**: Negative (0), Neutral (1), Positive (2)
- **Max Length**: 512 tokens
- **Batch Size**: 32 (adjustable based on GPU memory)

## Performance

- **Speed**: ~55-60 texts/second on Apple Silicon (MPS)
- **Accuracy**: Pre-trained model optimized for Twitter data
- **Memory**: Efficient batch processing to handle large datasets

## Device Support

The script automatically detects and uses the best available device:
- **MPS** (Metal Performance Shaders) for Apple Silicon Macs
- **CUDA** for NVIDIA GPUs
- **CPU** as fallback

## Configuration

You can adjust these parameters in `pretrained_twitter_model.py`:

```python
BATCH_SIZE = 32      # Batch size for processing
MAX_LENGTH = 512     # Maximum text length
```

## Example Output

```
Sentiment distribution:
neutral     506
negative    281
positive    213

Sentiment distribution percentage:
neutral     50.6
negative    28.1
positive    21.3
```

## File Structure

```
.
├── pretrained_twitter_model.py  # Main sentiment analysis script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── data/
    ├── covidvaccine.csv         # Input data
    └── covidvaccine_roberta_sentiment.csv  # Output results
```

## License

This project is part of the CISC 525 course work.

## Acknowledgments

- Model: [Cardiff NLP Twitter RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- Framework: [Hugging Face Transformers](https://huggingface.co/transformers/)
