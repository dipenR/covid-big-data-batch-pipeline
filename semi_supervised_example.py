"""
Example: Run semi-supervised sentiment training from Python.

For CLI usage, run:
  python semi_supervised_training.py -h
  python semi_supervised_training.py -l data/... -u data/...
"""

from semi_supervised_training import semi_supervised_training

if __name__ == "__main__":
    labeled_file = "data/covidvaccine_roberta_sentiment.csv"
    unlabeled_file = "data/covidvaccine.csv"

    model, tokenizer = semi_supervised_training(
        labeled_data_path=labeled_file,
        unlabeled_data_path=unlabeled_file,
        labeled_sample_size=1000,
        unlabeled_sample_size=20000,
        confidence_threshold=0.9,
        num_iterations=2,
        output_dir="./best_model",
    )

