# Reading Between the Lines: A dataset and a study on why some texts are tougher than others

This study is driven by the universal right to accessible information, see the full description of our project <https://idemproject.eu/en>, which aims at Text Simplification (TS) while leveraging strategies from both computational and translation studies. Particularly, intralingual translation, such as Diastratic Translation, focuses on shifting from Standard English to Easy-to-Read (E2R) English, making information accessible to audiences with reading difficulties, including people with disabilities and low literacy levels.

This repository contains a novel dataset, which is driven by taxonomy for sentence-level TS tasks. The dataset is located in the dataset/ folder. Unlike previous resources like [WikiLarge](https://github.com/XingxingZhang/dress) and [ASSET](https://github.com/facebookresearch/asset), which emphasize word-level or predefined operations, this study focuses on **why** corrections are needed by providing annotations for lexical, syntactic, and semantic changes. The dataset itself stems from diverse public services in Scotland:

<table>
  <thead>
    <tr>
      <th rowspan="2">Source</th>
      <th rowspan="2"># Texts</th>
      <th colspan="3">Complex</th>
      <th colspan="3">Simple</th>
    </tr>
    <tr>
      <th># Words</th>
      <th># Sentences</th>
      <th>IQR</th>
      <th># Words</th>
      <th># Sentences</th>
      <th>IQR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Health</td>
      <td>21</td>
      <td>183,677</td>
      <td>7,258</td>
      <td>(15.0–31.0)</td>
      <td>30,253</td>
      <td>1,519</td>
      <td>(10.0–21.0)</td>
    </tr>
    <tr>
      <td>Public Info</td>
      <td>4</td>
      <td>12,217</td>
      <td>527</td>
      <td>(12.0–30.5)</td>
      <td>3,378</td>
      <td>217</td>
      <td>(9.0–18.0)</td>
    </tr>
    <tr>
      <td>Politics</td>
      <td>9</td>
      <td>113,412</td>
      <td>4,824</td>
      <td>(15.0–29.0)</td>
      <td>12,474</td>
      <td>832</td>
      <td>(9.0–17.0)</td>
    </tr>
    <tr>
      <td>Data Selection</td>
      <td>–</td>
      <td>4,166</td>
      <td>155</td>
      <td>(12–27)</td>
      <td>3,259</td>
      <td>161</td>
      <td>(9–20)</td>
    </tr>
  </tbody>
</table>

---

The study's contributions include (1) an extended taxonomy of text simplification strategies that integrates insights from translation studies, (2) a corpus of complex and simplified texts sourced from public services in Scotland, (3) experiments using transformer-based models to predict simplification strategies, and (4) the use of Explainable AI (XAI) techniques, such as Integrated Gradients, to interpret model predictions. 



### **Text Simplification Macro-Strategies**
an extended taxonomy of text simplification strategies that integrates insights from translation studies

| **Macro-Strategy**         | **Strategies**                                                                                                                                                                                                                                   |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Transcription**           | No simplification needed.                                                                                                                                                                                                                     |
| **Synonymy**                | - **Pragmatic**: Acronyms spelled out; Proper names to common names; Contextual synonyms made explicit. <br>- **Semantic**: Hypernyms; Hyponyms; Stereotypes. <br>- **Grammatical**: Negative to positive sentences; Passive to active sentences; Pronouns to referents; Tenses simplified. |
| **Explanation**             | Words given for known; Expressions given for known; Tropes explained; Schemes explained; Deixis clarified; Hidden grammar made explicit; Hidden concepts made explicit.                                                                       |
| **Syntactic Changes**       | Word → Group; Word → Clause; Word → Sentence; Group → Word; Group → Clause; Group → Sentence; Clause → Word; Clause → Group; Clause → Sentence; Sentence → Word; Sentence → Group; Sentence → Clause.                                         |
| **Transposition**           | Nouns for things, animals, or people; Verbs for actions; Adjectives for nouns; Adverbs for verbs.                                                                                                                                             |
| **Modulation**              | Text-level linearity; Sentence-level linearity: Chronological order of clauses; Logical order of complements.                                                                                                                                 |
| **Anaphora**                | Repetition replaces synonyms.                                                                                                                                                                                                                |
| **Omission**                | Useless elements: Nouns; Verbs; Complements; Sentences. Rhetorical constructs; Diamesic elements.                                                                                                                                            |
| **Illocutionary Change**    | Implicit meaning made explicit.                                                                                                                                                                                                              |
| **Compression**             | Grammatical constructs simplified; Rhetorical constructs simplified.                                                                                                                                                                         |

---


### Using the simplification stratigies Classification Model


The annotated texts and necessary resources for training and evaluation are stored in the directory `texts/`.

### **Training the Model**
To train the model, run the following command:

```bash
python train.py -m <PLM> -l <checkpoint_dir> -i <train_file> --weights_file <class_weights_file> [hyperparameters]
```

#### **Required Parameters**:
- `-m`, `--model_name`: <PLM> The name of the pre-trained language model to use (e.g., bert-base-multilingual-cased, roberta-base).
- `-l`, `--local`: Directory to save model checkpoints and logs.
- `-i`, `--train_file`: Path to the training dataset (e.g., `texts/train.csv`).
- `--weights_file`: Path to the file containing class weights (e.g., `texts/class_weights.txt`).

#### **Optional Hyperparameters**:
- `-e`, `--epochs`: Number of training epochs (default: `4`).
- `--batch_size`: Batch size for training (default: `8`).
- `--learning_rate`: Learning rate for the optimizer (default: `1e-5`).
- `--weight_decay`: Weight decay (L2 regularization) for the optimizer (default: `0.01`).
- `--eval_steps`: Number of steps between evaluations (default: `1000`).
- `--fp16`: Use mixed-precision training for faster execution and reduced memory usage (flag; no value needed).
- `--max_grad_norm`: Maximum gradient norm for clipping (default: `1.0`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients before updating model weights (default: `1`).
- `--save_total_limit`: Maximum number of checkpoints to keep (default: `2`).
- `--logging_dir`: Directory to store logs (default: `./results/logs`).
- `--evaluation_strategy`: Evaluation strategy (`epoch`, `steps`, or `no`; default: `epoch`).
- `--save_strategy`: Strategy for saving checkpoints (`epoch`, `steps`, or `no`; default: `epoch`).

---
Available Pre-Trained Language Models (PLMs)

We recommend experimenting with widely-used transformer models like BERT, RoBERTa, and mBERT
#### **Example Training Command**
```bash
python train.py \
    -m bert-base-multilingual-cased \
    -l ./results \
    -i texts/train.csv \
    --weights_file texts/class_weights.txt \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --eval_steps 500 \
    --fp16 \
    --max_grad_norm 1.0
```

To view the full list of parameters and their descriptions:
```bash
python train.py -h
```

---

### **Evaluating the Model**
To evaluate the trained model, run the following command:

```bash
python test.py -m <checkpoint_dir> -t <test_file>
```

#### **Required Parameters**:
- `-m`, `--model_dir`: Path to the saved model directory (e.g., `./results`).
- `-t`, `--test_file`: Path to the test dataset (e.g., `texts/test.csv`).

---

#### **Example Evaluation Command**
```bash
python test.py \
    -m ./results \
    -t texts/test.csv
```

---

### **Splitting the Data**
Before training and evaluation, you can split your dataset into training and testing subsets using `split_data.py`:

```bash
python split_data.py \
    -i texts/annotated.csv \
    --output_train texts/train.csv \
    --output_test texts/test.csv \
    --output_weights texts/class_weights.txt \
    --test_size 0.2
```

#### **Parameters for `split_data.py`**:
- `-i`, `--inputfile`: Path to the annotated dataset.
- `--output_train`: Path to save the training dataset.
- `--output_test`: Path to save the test dataset.
- `--output_weights`: Path to save the class weights file.
- `--test_size`: Proportion of data to use for testing (default: `0.2`).

---

### **Note**
**Class Weights**:
   - Ensure the `class_weights.txt` is generated using `split_data.py`. This ensures the model effectively handles class imbalances.


### Citation
The dataset and the script are fully described in our paper:
**Reading Between the Lines: A Dataset and a Study on Why Some Texts Are Tougher Than Others**,
*Nouran Khallaf, Carlo Eugeni, and Serge Sharoff*  
Presented at **Writing Aids at the Crossroads of AI, Cognitive Science, and NLP (WR-AI-CogS)**, COLING 2025, Abu Dhabi.  
[arXiv:2501.01796](https://arxiv.org/abs/2501.01796)

```
@inproceedings{khallaf2025readinglinesdatasetstudy,
  title={Reading Between the Lines: A dataset and a study on why some texts are tougher than others},
  author={Nouran Khallaf and Carlo Eugeni and Serge Sharoff},
  booktitle={Writing Aids at the Crossroads of AI, Cognitive Science and NLP WR-AI-CogS, at COLING'2025},
  address={Abu Dhabi},
  year={2025},
  eprint={2501.01796},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2501.01796}
}
```



