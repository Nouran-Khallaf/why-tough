# Reading Between the Lines: A dataset and a study on why some texts are tougher than others

This study is driven by the universal right to accessible information, see the full description of our project <https://idemproject.eu/en>, which aims at Text Simplification (TS) while leveraging strategies from both computational and translation studies. Particularly, intralingual translation, such as Diastratic Translation, focuses on shifting from Standard English to Easy-to-Read (E2R) English, making information accessible to audiences with reading difficulties, including people with disabilities and low literacy levels.

This repository contains a novel dataset, which is driven by taxonomy for sentence-level TS tasks. Unlike previous resources like [WikiLarge](https://github.com/XingxingZhang/dress) and [ASSET](https://github.com/facebookresearch/asset), which emphasize word-level or predefined operations, this study focuses on **why** corrections are needed by providing annotations for lexical, syntactic, and semantic changes. The dataset itself stems from diverse public services in Scotland:

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

The annotated texts are in the directory */texts/*.

To train the model, run:

    train.py PLM checkpoit /texts/annotated.csv [hyperparameters]

As the PLMs we used the traditional selection of BERT and RoBERTa models.


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



