#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Function to load class weights
def load_class_weights(weights_file):
    class_weights = []
    with open(weights_file, 'r') as f:
        next(f)  
        for line in f:
            _, _, weight = line.strip().split(',')
            class_weights.append(float(weight))
    return torch.tensor(class_weights)

# Prepare dataset for training
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

# Custom Trainer to use class weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Accept **kwargs to handle extra arguments
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Train the model
def train_model(model_name, train_file, eval_file, weights_file, training_args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load datasets
    train_dataset = prepare_dataset(train_file, tokenizer)
    eval_dataset = prepare_dataset(eval_file, tokenizer)

    # Load class weights
    class_weights = load_class_weights(weights_file)

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(class_weights)
    )

    # Define Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda pred: {
            "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1)),
            "precision_recall_fscore": precision_recall_fscore_support(pred.label_ids, pred.predictions.argmax(-1), average='weighted')[:3]
        }
    )
    trainer.class_weights = class_weights

    # Train model
    trainer.train()
    print("Training complete.")

    # Evaluate once to avoid redundant calls
    eval_results = trainer.evaluate()
    eval_f1 = eval_results.get("eval_f1", 0)  # Default to 0 if missing

    # Save only if the new model has a better F1-score
    if eval_f1 > train_model.best_f1:
        train_model.best_f1 = eval_f1
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f" New best model saved to {training_args.output_dir} with F1-score: {train_model.best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Transformer Model for Complexity Classification")
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-multilingual-cased', help='Model name according to HuggingFace transformers')
    parser.add_argument('-l', '--local', type=str, default=None, help='Directory for the local model')
    parser.add_argument('-p', '--projectname', type=str, default=None, help='Project name for tracking (optional)')
    parser.add_argument('-i', '--train_file', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('-weights','--weights_file', type=str, required=True, help='Path to the class weights file')
    parser.add_argument('-out','--output_dir', type=str, default='./results', help='Directory to save the model')
    parser.add_argument('--sep', type=str, default=',', help='Separator for the dataset file')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the evaluation dataset')

    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=f"{args.output_dir}/logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=args.fp16,
        max_grad_norm=args.max_grad_norm,
        report_to="none" 
    )

    train_model(
    model_name=args.model_name,
    train_file=args.train_file,
    eval_file=args.eval_file, 
    weights_file=args.weights_file,
    training_args=training_args
    )

