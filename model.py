#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Define a Custom Trainer to Incorporate Class Weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_model(model_name, train_dataset, val_dataset, training_args):
    # Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda pred: {
            "accuracy": accuracy_score(pred.label_ids, pred.predictions.argmax(-1))
        },
    )

    # Train and Save Model
    trainer.class_weights = class_weights
    trainer.train()

    # Save model and tokenizer locally
    model.save_pretrained(f"./{model_name}_complex")
    tokenizer.save_pretrained(f"./{model_name}_complex")

    print(f"Model saved locally under: ./{model_name}_complex")

def prepare_datasets(dataset_path, sep, valsplit):
    # Load Data
    data = pd.read_csv(dataset_path, sep=sep)
    data = data.dropna(subset=['Complex', 'Strategy'])
    data['Typology'] = data['Strategy'].str.replace("\xa0", "").str.replace(" ", "")

    # Map Typologies to Macro-Categories
    macro_to_original = {
        'Transcript': ['Transcript'],
        'Substitution': [
            'Synonymy', 'Anaph', 'SemStereo', 'SemStere', 'SemHype', 'Hypernyms', 'Hyponyms', 'Acronyms',
            'SemHypeÊ', 'SemStereÊ', 'SemHypo','SemHype\xa0','SemStere\xa0'
        ],
        'Grammatical Adjustments': [
            'GraPron', 'GraTens', 'GraSim', 'GraPass', 'Negations', 'Pronouns', 'PassiveVoice', 'GraPassÊ','GraPass\xa0'
        ],
        'Explanation': [
            'ExplWor', 'WordExpl', 'ExplCont', 'ExplExpr', 'ExpExp', 'Tropes', 'Schemes', 'Deixis', 'HidGram',
            'HidCont', 'WorExp', 'HidGra', 'HidCon', 'PraCont', 'PraExp', 'PraExpÊ', 'PrAcron', 'PraProp'
        ],
        'Syntactic Changes': [
            'SynChange', 'Clause2Word', 'WordsOrder', 'GroupOrder', 'LinearOrderSen', 'LinearOrderCla',
            'SynW2G', 'SynG2W', 'SynC2W', 'SynG2T', 'SynW2C', 'SynW2S', 'SynC2S', 'SynG2C', 'SynG2WÊ',
            'SynW2SÊ', 'SynSem', 'SynG2W\xa0', 'SynW2S\xa0'
        ],
        'Transposition': ['TranspNoun', 'TraNou', 'TraVer', 'Trans'],
        'Modulation': [
            'ModInfo', 'ModInf', 'ClauseOrder', 'GroupOrder', 'WordOrder', 'ModClau', 'ModWord', 'ModGrou'
        ],
        'Omission': [
            'OmiSen', 'OmiWor', 'OmiClau', 'OmiRhet', 'OmiComp', 'OmiSubj', 'OmiSent', 'OmiVer', 'OmiRhe'
        ],
        'Simplification': [
            'SinGram', 'SinSem', 'SimGram', 'SinPrag', 'SimplifiedSyntax', 'SimplifiedTenses', 'S-V-O Structures'
        ],
        'Illocutionary Change': ['IllCh']
    }

    original_to_macro = {typology: macro for macro, typologies in macro_to_original.items() for typology in typologies}
    data['Typology'] = data['Strategy'].replace(original_to_macro)

    # Encode Typologies
    label_encoder = LabelEncoder()
    data['Typology Encoded'] = label_encoder.fit_transform(data['Typology'])

    # Calculate class weights
    class_counts = data['Typology Encoded'].value_counts()
    class_weights = torch.tensor([1.0 / count * len(data) / 2.0 for count in class_counts])

    # Split Data
    train_size = int(len(data) * (1 - valsplit))
    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    def tokenize_function(texts):
        return tokenizer(list(texts), padding=True, truncation=True, max_length=512)

    train_encodings = tokenize_function(train_data['Complex'])
    val_encodings = tokenize_function(val_data['Complex'])

    train_labels = torch.tensor(train_data['Typology Encoded'].values, dtype=torch.long)
    val_labels = torch.tensor(val_data['Typology Encoded'].values, dtype=torch.long)

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

    train_dataset = Dataset(train_encodings, train_labels)
    val_dataset = Dataset(val_encodings, val_labels)

    return train_dataset, val_dataset

# Main Script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A Transformer Model for Complexity Classification")
    parser.add_argument('-m', '--mname', type=str, default='bert-base-multilingual-cased', help='Model name according to HuggingFace transformers')
    parser.add_argument('-l', '--local', type=str, default=None, help='Directory for the local model')
    parser.add_argument('-p', '--projectname', type=str, default=None, help='Project name for tracking (optional)')
    parser.add_argument('-i', '--inputfile', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('-t', '--testfile', type=str, default=None, help='Path to the testing dataset')
    parser.add_argument('--sep', type=str, default=',', help='Separator for the dataset file')

    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Evaluation steps')
    parser.add_argument('--valsplit', type=float, default=0.2, help='Validation split fraction')

    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(args.inputfile, args.sep, args.valsplit)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.local if args.local else "./results",
        evaluation_strategy="steps",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"./logs",
        logging_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True
    )

    train_model(args.mname, train_dataset, val_dataset, training_args)
