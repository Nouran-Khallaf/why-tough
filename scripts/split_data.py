#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typology_mapping import macro_to_original  # Import typology mapping from external file

# Function to prepare and split data into train and test sets
def prepare_and_split_data(input_file, output_train, output_test, output_weights, test_size, sep):
    # Load the dataset
    data = pd.read_csv(input_file, sep=sep)

    # Drop empty columns
    data = data.dropna(how='all', axis=1)

    # Drop rows with missing values in required columns
    data = data.dropna(subset=['Complex', 'Strategy'])

    # Clean up the Strategy column
    data['Strategy'] = data['Strategy'].str.replace("\xa0", "").str.replace(" ", "")

    # Map Typologies to Macro-Categories
    original_to_macro = {typology: macro for macro, typologies in macro_to_original.items() for typology in typologies}
    data['Typology'] = data['Strategy'].replace(original_to_macro)

    # Encode Typologies
    label_encoder = LabelEncoder()
    data['Typology Encoded'] = label_encoder.fit_transform(data['Typology'])

    # Calculate class weights
    class_counts = data['Typology Encoded'].value_counts()
    class_weights = torch.tensor([1.0 / count * len(data) / 2.0 for count in class_counts])
    print(f"Class Weights: {class_weights.tolist()}")

    # Save class weights with class names and encoded values to a file
    with open(output_weights, 'w') as f:
        f.write("Class,Encoded,Weight\n")
        for cls, weight in zip(class_counts.index, class_weights.tolist()):
            class_name = label_encoder.inverse_transform([cls])[0]
            f.write(f"{class_name},{cls},{weight}\n")
    print(f"Class weights saved to: {output_weights}")

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42, stratify=data['Typology Encoded'])

    # Save the train and test sets
    train_data.to_csv(output_train, index=False)
    test_data.to_csv(output_test, index=False)

    print(f"Train and test data saved: \nTrain -> {output_train}\nTest -> {output_test}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and split dataset into training and testing sets")
    parser.add_argument('-i', '--inputfile', type=str, required=True, help='Path to the input dataset (CSV)')
    parser.add_argument('--output_train', type=str, required=True, help='Path to save the training dataset')
    parser.add_argument('--output_test', type=str, required=True, help='Path to save the testing dataset')
    parser.add_argument('--output_weights', type=str, required=True, help='Path to save the class weights')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to use as test set (default: 0.2)')
    parser.add_argument('--sep', type=str, default=',', help='Separator for the dataset file (default: ",")')

    args = parser.parse_args()

    prepare_and_split_data(args.inputfile, args.output_train, args.output_test, args.output_weights, args.test_size, args.sep)
