#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

# Function to evaluate the model
def evaluate_model(model_path, test_file, sep):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Load test data
    data = pd.read_csv(test_file, sep=sep)
    texts = data['Complex']
    labels = data['Typology Encoded']

    # Tokenize the test data
    encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Run the model and get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a Saved Transformer Model")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('-t', '--test_file', type=str, required=True, help='Path to the test dataset')
    parser.add_argument('--sep', type=str, default=',', help='Separator for the dataset file')

    args = parser.parse_args()

    metrics = evaluate_model(args.model_path, args.test_file, args.sep)

    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
