# ViT_for_affinity

ViT_for_affinity is a vision transformer applied to the problem of predicting affinity of protein-ligand complexes using 3D structures as input data.
This repository contains the official implementation of the model described in:

> **Application of vision transformers to protein-ligand affinity prediction.**
> Authors: Poziemski J, Siedlecki P.
> DOI: 10.26434/chemrxiv-2025-qcxq1

---

## Overview

The ViT model predicts protein-ligand binding affinities using Vision transformer architecture based on 3D Voxels.


![Model Architecture](/images/ViT_arch.png)

---

## Repository Structure

```bash
├── files/   # annotation files
├── sample_data/  
├── utils.py         
├── featurizer.py # Script to generate input data         
├── train.py     # Script to train data 
├── grid.py     
├── predict.py     # Script for data prediction
├── vit_model.py     # Model architecture class
├── README.md           
└── VIT_base_model.pth  # Base model used for analysis in main paper
```


## Requirements

- Python 3.9.19
- rdkit 2023.9.6
- torch 2.3.1
- einops 0.8.0
- mdanalysis 2.7.0
- pandas 
- BIOpython 
- dssp package


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
python scoring_evaluation.py --model VIT_base_model.pth  --grid_dir [path to directory with perpared grids ]  --coreset_2016  [coreset 2016 annotation file] --coreset_2013 [coreset 2013 annotation file]
```

## Citation

Poziemski J, Siedlecki P. Application of vision transformers to protein-ligand affinity prediction. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-qcxq1

## Licence

CC BY 4.0 
