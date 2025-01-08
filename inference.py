#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from numpy import load
from sklearn.preprocessing import LabelEncoder

def predict(model_path, input_texts, label_classes_path):
    # Load tokenizer, model, and label classes
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    label_classes = load(label_classes_path)  # Load label classes (e.g., classes and encodings)

    # Tokenize input texts
    encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # Run the model and get predictions
    with torch.no_grad():
        outputs = model(**encodings)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Decode the predicted labels using the loaded label classes
    decoded_predictions = [label_classes[int(pred)] for pred in predictions]

    return decoded_predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using a trained Transformer model")
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to the saved model directory')
    parser.add_argument('-i', '--input_texts', type=str, nargs='+', required=True, help='List of texts to classify')
    parser.add_argument('-l', '--label_classes_path', type=str, required=True, help='Path to the label classes file (e.g., label_classes.npy)')

    args = parser.parse_args()

    predictions = predict(args.model_path, args.input_texts, args.label_classes_path)

    print("Predictions:")
    for text, label in zip(args.input_texts, predictions):
        print(f"Text: {text}\nPredicted Label: {label}\n")
