# VIT_for_affinity


This repository contains the official implementation of the model described in:

> **[Paper Title]**
> Authors: [Full names] 
> Published at: [Conference/Journal Name, Year]
> DOI / arXiv: [Link]

---

## 🚀 Overview

This model predicts protein-ligand binding affinities using a hybrid graph neural network architecture.

If applicable, include a diagram or conceptual figure from the paper.

---

## Repository Structure

```bash
├── files/                 # Main source code
│   ├── model.py         # Model architecture
│   └── utils.py         # Utility functions
├── sample_data/                # Sample input data or download scripts
├── utils.py         # Pretrained model weights
├── featurizer.py           # YAML/JSON config files
├── train.py     # Python dependencies
├── grid.py     # Python dependencies
├── predict.py     # Python dependencies
├── vit_model.py     # Python dependencies
├── README.md            # Project readme
└── VIT_base_model.pth 
```


## Requirements

Python 3.9.19
rdkit 2023.9.6
torch  2.3.1
einops  0.8.0
mdanalysis 2.7.0 

other:
pandas, 
BIOpython, 
dssp package


## Usage

### Data preparation


```bash
python featurizer.py --input_dir [directory_with_pdbbind_like_structure] --output_dir [path_to_output directory]
```

### Training

To simply reproduce model first you need to create complex grids by command

```bash
python train.py --grid_dir [path to perpared grids directory] --coreset_2016  [coreset 2016 annotation file ] --coreset_2013 [coreset 2013 annotation file] --train_data  [train data annotation file] --valid_data [validation data annotation file]
```

Script required annotation files:

All scripts in this repository assumes that annotations file contains 2 columns:
first columns contains id, typically pdb code
second columns conatains normalized affinity value. You can see examples is 'files' directory. 

### Score complexes

```bash
python predict.py --model VIT_base_model.pth  --grid_dir [path to directory with perpared grids ] --output_file results.csv
```

### Evaluation

You can simply evaluate out model on coreset2013 and coreset2016 by:
```bash
python scoring_evaluation.py --model VIT_base_model.pth  --grid_dir [path to directory with perpared grids ]  --coreset_2016  [coreset 2016 annotation file ] --coreset_2013 [coreset 2013 annotation file]
```

## Citation

Poziemski J, Siedlecki P. Application of vision transformers to protein-ligand affinity prediction. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-qcxq1

## Licence

CC BY 4.0 
