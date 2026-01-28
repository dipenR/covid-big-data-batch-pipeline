# Semi-Supervised Learning – How to Train

## 1. Data you need

- **Labeled CSV**: `text` and `sentiment` columns. Sentiment must be `negative`, `neutral`, or `positive`.  
  Example: `data/covidvaccine_roberta_sentiment.csv` (from `pretrained_twitter_model.py`).

- **Unlabeled CSV**: `text` column only.  
  Example: `data/covidvaccine.csv`.

## 2. Run from command line

```bash
# Activate your env (e.g. cisc525)
source cisc525/bin/activate

# Basic run
python semi_supervised_training.py \
  --labeled data/covidvaccine_roberta_sentiment.csv \
  --unlabeled data/covidvaccine.csv

# Full options
python semi_supervised_training.py \
  --labeled data/covidvaccine_roberta_sentiment.csv \
  --unlabeled data/covidvaccine.csv \
  --labeled-size 1000 \
  --unlabeled-size 20000 \
  --confidence 0.9 \
  --iterations 2 \
  --output-dir ./best_model
```

**Options:**

| Option | Short | Default | Meaning |
|--------|-------|---------|---------|
| `--labeled` | `-l` | required | Path to labeled CSV |
| `--unlabeled` | `-u` | required | Path to unlabeled CSV |
| `--labeled-size` | | all | Max number of labeled examples |
| `--unlabeled-size` | | 50000 | Max unlabeled examples used for pseudo-labeling |
| `--confidence` | `-c` | 0.9 | Min confidence to keep a pseudo-label |
| `--iterations` | `-i` | 2 | Self-training iterations |
| `--output-dir` | `-o` | `./best_model` | Where to save the best model |

## 3. Run from Python

```python
from semi_supervised_training import semi_supervised_training

model, tokenizer = semi_supervised_training(
    labeled_data_path="data/covidvaccine_roberta_sentiment.csv",
    unlabeled_data_path="data/covidvaccine.csv",
    labeled_sample_size=1000,
    unlabeled_sample_size=20000,
    confidence_threshold=0.9,
    num_iterations=2,
    output_dir="./best_model",
)
```

## 4. Use the trained model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./best_model")
tokenizer = AutoTokenizer.from_pretrained("./best_model")

# Run inference as in pretrained_twitter_model.py
```

## 5. Quick test (small data)

```bash
python semi_supervised_training.py \
  -l data/covidvaccine_roberta_sentiment.csv \
  -u data/covidvaccine.csv \
  --labeled-size 500 \
  --unlabeled-size 5000 \
  -c 0.9 -i 1 -o ./best_model_test
```

## 6. Workflow summary

1. Build labeled data (e.g. run `pretrained_twitter_model.py` → `*_roberta_sentiment.csv`).
2. Keep raw tweets as unlabeled CSV (`text` only).
3. Run `semi_supervised_training.py` with `--labeled` and `--unlabeled`.
4. Load `./best_model` (or your `--output-dir`) for inference.
