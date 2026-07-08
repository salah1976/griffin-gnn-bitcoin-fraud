
# Griffin: A Heterogeneous Graph Neural Network for Realistic Bitcoin Fraud Detection

Official implementation of the paper:

> **Griffin: A Heterogeneous Graph Neural Network for Realistic Bitcoin Fraud Detection**

Submitted to the *Journal of Artificial Intelligence Research (JAIR)*.

---

## Overview

Graph Neural Networks (GNNs) have recently shown promising performance for blockchain fraud detection by exploiting the relational structure of Bitcoin transactions. However, existing evaluations often overlook how persistent graph connectivity influences predictive performance under realistic deployment conditions.

This repository provides the official implementation of **Griffin**, a heterogeneous Graph Neural Network designed for transaction-level Bitcoin fraud detection on heterogeneous Bitcoin UTXO graphs.

The framework includes:

- heterogeneous graph construction;
- GraphSAGE-based representation learning;
- confidence-weighted focal loss;
- temporal evaluation protocol;
- structural exposure analysis;
- comparison with a LightGBM baseline;
- explainability using SHAP and UMAP.

---

## Repository Structure

```
griffin-gnn-bitcoin-fraud/

├── README.md
├── LICENSE
├── requirements.txt
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── structural_exposure.py
│   ├── lightgbm_baseline.py
│   ├── shap_analysis.py
│   ├── umap_visualization.py
│   └── ...
│
├── checkpoints/
│   └── griffin_best.pt
│
├── figures/
│
├── tables/
│
└── docs/
```

---

## Dataset

The experiments use the **BitFraud** benchmark.

The dataset is **not stored in this repository** because of its size.

It is publicly available from Zenodo:

**DOI**

 https://zenodo.org/uploads/17428642

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/griffin-gnn-bitcoin-fraud.git

cd griffin-gnn-bitcoin-fraud
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train Griffin

```bash
python scripts/train.py
```

---

## Evaluation

Evaluate the trained model

```bash
python scripts/evaluate.py
```

---

## Structural Exposure Analysis

Run the structural connectivity analysis

```bash
python scripts/structural_exposure.py
```

---

## LightGBM Baseline

Train and evaluate the feature-based baseline

```bash
python scripts/lightgbm_baseline.py
```

---

## Explainability

Generate SHAP explanations

```bash
python scripts/shap_analysis.py
```

Generate UMAP visualizations

```bash
python scripts/umap_visualization.py
```

---

## Reproducibility

This repository has been developed to support the reproducibility of the experimental results reported in the accompanying paper.

The complete reproducibility package includes:

- source code;
- trained models;
- experimental scripts;
- figure generation scripts;
- table generation scripts;
- configuration files;
- documentation.

The BitFraud benchmark is distributed separately through Zenodo.

---

## Citation

If you use this repository, please cite:

```bibtex
@article{bouchama2026griffin,
  title={Griffin: A Heterogeneous Graph Neural Network for Realistic Bitcoin Fraud Detection},
  author={Bouchama, Salah-Eddine and Ouchani, Samir and Bouarfa, Hafida},
  journal={Journal of Artificial Intelligence Research},
  year={2026},
  note={Under review}
}
```

---

## License

This project is released under the MIT License.

---

## Contact

**Salah-Eddine Bouchama**

LRDSI Laboratory

Blida 1 University

Blida, Algeria

Email: BOUCHAMA_SalahEddine@univ-blida.dz
