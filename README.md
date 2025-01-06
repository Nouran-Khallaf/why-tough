# Reading Between the Lines: A dataset and a study on why some texts are tougher than others

This study is driven by the universal right to accessible information, see the full description of our project <https://idemproject.eu/en>, which aims at Text Simplification (TS) while leveraging strategies from both computational and translation studies. Particularly, intralingual translation, such as Diastratic Translation, focuses on shifting from Standard English to Easy-to-Read (E2R) English, making information accessible to audiences with reading difficulties, including people with disabilities and low literacy levels.

This repository contains a novel dataset, which is driven by taxonomy for sentence-level TS tasks. Unlike previous resources like [WikiLarge](https://github.com/XingxingZhang/dress) and [ASSET](https://github.com/facebookresearch/asset), which emphasize word-level or predefined operations, this study focuses on **why** corrections are needed by providing annotations for lexical, syntactic, and semantic changes. The dataset itself stems from diverse public services in Scotland:
|  |  |  |  |  |  |  |  |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Source** | **\#Texts** | **Complex** |  |  | **Simple** |  |  |
|  |  | \#Words | \#Sentences | IQR | \#Words | \#Sentences | IQR |
| Health | 21 | 183677 | 7258 | (15.0-31.0) | 30253 | 1519 | (10.0-21.0) |
| Public info | 4 | 12217 | 527 | (12.0-30.5) | 3378 | 217 | (9.0-18.0) |
| Politics | 9 | 113412 | 4824 | (15.0-29.0) | 12474 | 832 | (9.0-17.0) |
| Data selection | – | 4166 | 155 | (12-27) | 3259 | 161 | (9-20) |

The study's contributions include (1) an extended taxonomy of text simplification strategies that integrates insights from translation studies, (2) a corpus of complex and simplified texts sourced from public services in Scotland, (3) experiments using transformer-based models to predict simplification strategies, and (4) the use of Explainable AI (XAI) techniques, such as Integrated Gradients, to interpret model predictions. 

The annotated texts are in the directory */texts/*.

To train the model, run:

    train.py PLM checkpoit /texts/annotated.csv [hyperparameters]

As the PLMs we used the traditional selection of BERT and RoBERTa models.

The dataset and the script are fully described in our paper:

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



