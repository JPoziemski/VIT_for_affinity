# ViT for affinity

ViT_for_affinity is a vision transformer architecture applied to the problem of predicting affinity of protein-ligand complexes using 3D structures as input data.

---

## Overview

Vision Transformers (ViTs) have found wide application in various areas of computer vision, including image classification, object detection and image generation; typically achieving results that outperform traditional convolutional architectures. Due to their strong ability to model global dependencies, ViTs have also been successfully applied to advanced tasks such as medical image analysis or video analysis. 
ViTs can also be successfully applied to the problems related to predicting protein-ligand affinity, a highly important challenge in computer aided drug design (CADD). The affinity is defined here as the strength with which a small-molecule ligand binds to a molecular target (protein), expressed as a real number. 
We use a set of 3D structures of protein-ligand complexes and transform them into a set of related tokens which serve as the basis for the learning process. We evaluate our method against the best models to date, using two independent datasets of experimental measurements.

![Model Architecture](/images/ViT_arch.png)

Despite the problem’s complexity (small dataset, high sparsity, activity cliffs), ViT can be effectively applied to protein-ligand affinity prediction. XAI results indicate ViT learns from valid interaction patterns and focuses on relevant molecular information. The ViT architecture presented here has potential for further optimization, potentially enhancing performance.
Our results provide a foundation for using ViT in medically relevant problems hindered by data scarcity, such as ligand-RNA interactions. More in:

> **Application of vision transformers to protein-ligand affinity prediction.**
> Authors: Poziemski J, Siedlecki P.
> [https://doi.org/10.26434/chemrxiv-2025-qcxq1](https://doi.org/10.26434/chemrxiv-2025-qcxq1)
---

## Repository Structure

```bash
├── files/             # Annotation files
├── sample_data/  
├── utils.py         
├── featurizer.py      # Script to generate input data         
├── train.py           # Script to train data 
├── grid.py     
├── predict.py         # Script for data prediction
├── vit_model.py       # Model architecture class
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
- DSSP package


## Usage

### Data preparation

Take protein–ligand 3D structures as input and produce a set of feature grids for each complex.

```bash
python featurizer.py --input_dir [dataset_structure] --output_dir [path_to_output_directory]
```

```
example_dataset_structure/
└── [PDB_ID]/                  # one folder per complex, named by its PDB ID
    ├── [PDB_ID]_protein.pdb   # receptor file (protein only)
    ├── [PDB_ID]_ligand.mol2   # ligand file (mol2/sdf)
```


### Training

To reproduce the model, the protein-ligand grids are used, along with train, validation and test split annotation files containing affinity measurements.

```bash
python train.py --grid_dir [path_to_feature_grids_directory] --train_data [train_data_annotation_file] --valid_data [validation_data_annotation_file] --casf_2016 [casf_2016_annotation_file] --coreset_2013 [coreset_2013_annotation_file] 
```

```
Required annotation files:
All scripts in this repository assume that annotation files contains 2 columns:
- first columns contains the ID, typically pdb code (PDBID)
- second columns contains the normalized affinity value (negative base-10 logarithms of molar concentrations). See 'files' directory. 
```

### Predict affinity 

Once trained, the ViT_for_affinity model can be used to predict affinity for novel protein-ligand complexes

```bash
python predict.py --model [VIT_base_model.pth]  --grid_dir [path_to_feature_grids_directory] --output_file [predictions.csv]
```

### Evaluation

To reproduce the results reported in [https://doi.org/10.26434/chemrxiv-2025-qcxq1](https://doi.org/10.26434/chemrxiv-2025-qcxq1) on CASF_2016 and Coreset_2013 you can run the following command: 
```bash
python scoring_evaluation.py --model VIT_base_model.pth  --grid_dir [path to directory with perpared grids ]  --coreset_2016  [coreset 2016 annotation file] --coreset_2013 [coreset 2013 annotation file]
```

## Citation

Poziemski J, Siedlecki P. Application of vision transformers to protein-ligand affinity prediction. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-qcxq1

## Licence

CC BY 4.0 
