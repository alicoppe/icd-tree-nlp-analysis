# Clinical Text Embedding and ICD-10 Hierarchy NLP Analysis

This repository contains the source code and experiments for the project: **"Clinical Text Embedding and the Hierarchical Relationship between ICD-10 Classification Codes"**. The project evaluates how well text embedding models capture the hierarchical relationships inherent in ICD-10 codes.

## Project Overview

The study focuses on leveraging text embeddings generated by Base BERT, Clinical BERT, and Random BERT models to assess their alignment with the hierarchical structure of ICD-10 codes. We compared pairwise embedding distances with ICD tree-based distances using Spearman correlations.

Key contributions:
- Implemented a directed acyclic graph (DAG) to represent ICD-10 hierarchies.
- Evaluated embedding models' abilities to capture hierarchical relationships through correlation metrics.
- Proposed depth-weighted optimization to address clinical differences in ICD depth structure, and relatedness amongst conditions.

## Repository Structure

- `scripts/`: Contains reusable modules and classes, including:
  - **`icd_tree.py`**: Implements the DAG representation of ICD-10, with methods for calculating distances and visualizing paths.
  - **`bert_wrapper.py`**: Creates a wrapper class for the BERT model that calculates the embeddings based on the specified model type (base model, clinical, and random).
  - **`data.py`**: Contains helper functions for modifying the data, such as creating batches, local shuffling, and Spearman correlation evaluation.
  - **`mc_searcg.py`**: Implements the Monte Carlo searching algorithm for the ICDTree depth weight optimization.
  - **`probe.py`**: Pytorch implementation of structural probe for evaluating if the hierarchical structure is encoded in the embeddings.
  - **`loss.py`**: L1 loss function for the training of the structural probe.
  - **`regimen.py`**: Implements training regimen of the structural probe, along with prediction.
  
- `notebooks/`: Jupyter notebooks for running experiments and visualizing results:
  - **`ICD_tree_tester.ipynb`**: Tests the ICD tree structure and calculates Spearman correlations for embedding distances.
  - **`weight_adjustment_tester.ipynb`**: Optimizes depth weights using Monte Carlo search and evaluates its impact on correlations.
  - **`full_probing_approach.ipynb`**: Implements a structural probing approach inspired by the work of Hewitt & Manning (2019), analyzing the structural geometry of ICD embeddings.
- `README.md`: Project overview and usage instructions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alicoppe/icd-tree-nlp-analysis.git
