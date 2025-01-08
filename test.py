#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Prepare dataset for evaluation
def prepare_dataset(file_path, tokenizer):
    data = pd.read_csv(file_path)
    encodings = tokenizer(list(data['Complex']), padding=True, truncation=True, max_length=512)
    labels = torch.tensor(data['Typology Encoded'].values, dtype=torch.long)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = self.labels[idx]
            return item

    return Dataset(encodings, labels)

# Evaluate the model
def evaluate_model(saved_model_path, eval_file, output_dir):
    # Load the saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)

    # Prepare evaluation dataset
    eval_dataset = prepare_dataset(eval_file, tokenizer)

    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=8,
        report_to="none" 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # Evaluate the model
    eval_metrics = trainer.evaluate()
    print("Evaluation Metrics:", eval_metrics)

    # Make predictions
    predictions = trainer.predict(eval_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)

    # Generate classification report
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(f"./label_classes.npy", allow_pickle=True)

    target_names = label_encoder.classes_
    true_labels = eval_dataset.labels.numpy()

    print("Classification Report:")
    print(classification_report(
        true_labels,
        predicted_labels,
        target_names=target_names,
        zero_division=0
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Transformer Model for Complexity Classification")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the evaluation dataset')
    parser.add_argument('--output_dir', type=str, default='./eval_results', help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_model(
        saved_model_path=args.model_path,
        eval_file=args.eval_file,
        output_dir=args.output_dir
    )
