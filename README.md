# PADI: Post-Anomaly Detection Inference

**PADI** is a novel framework that equips a trained and frozen Deep SVDD detector with statistically valid inference by leveraging the Selective Inference framework. 

Despite its empirical success, anomaly decisions produced by Deep SVDD are typically made solely based on anomaly scores without rigorous statistical guarantees. To address the fundamental double-dipping issue arising from conducting inference after anomaly detection, PADI performs inference conditional on the event that a test instance is identified as anomalous by Deep SVDD. Based on this formulation, PADI derives valid selective $p$-values that quantify the statistical significance of the detected anomaly, theoretically establishing control of the false positive rate (FPR) at a user-specified significance level $\alpha$.

![Figure 1: PADI Overview](figures/figure%201.png)

## Contributions & Key Features
- **Statistically Valid Inference:** Derives valid selective $p$-values that provably control the FPR, addressing the double-dipping issue.
- **Post Hoc Operation:** Operates in a post hoc manner and does not require retraining or modifying the underlying anomaly detector.
- **Broad Applicability:** Directly applicable to both hard-boundary and soft-boundary Deep SVDD variants, and naturally extends to Deep Semi-Supervised Anomaly Detection (Deep SAD).
- **GPU Acceleration:** Provides a GPU-accelerated implementation of PADI using custom Numba-CUDA kernels to improve computational efficiency for complex CNN architectures.

## Directory Structure

The codebase is logically split into two primary pipelines:

- `PADI_tabular/`: Pipeline for tabular data using fully connected neural networks (MLP).
- `PADI_image/`: Pipeline for image data using Convolutional Neural Networks (CNN) with operations like `Conv2d`, `BatchNorm2d`, `LeakyReLU`, and `MaxPool2d`.
- `util.py`: Shared core mathematical operations, including linear/quadratic inequality solvers, interval arithmetic, and high-precision truncated Gaussian integrations via `mpmath`.

## Requirements

Ensure you have Python $\ge$ 3.10 installed. You can install the required dependencies via pip:

```bash
pip install -r requirements.txt
```
*(Note: `numba` is required for the GPU-accelerated implementation of PADI on CNNs).*

## Quick Start (Examples)

Both pipelines include comprehensive Jupyter Notebook examples. 

### 1. Tabular Data Pipeline
Navigate to the tabular examples directory to evaluate the model on synthetic data:
```bash
cd PADI_tabular/examples

# Compute the selective p-value for a detected anomaly
jupyter notebook ex1_tab_pvalue.ipynb

# Validate the FPR control using normal reference samples
jupyter notebook ex2_tab_fpr.ipynb
```

### 2. Image (CNN) Data Pipeline
Navigate to the image examples directory to evaluate the model on synthetic image patches:
```bash
cd PADI_image/examples

# Compute the selective p-value for an anomalous image patch
jupyter notebook ex1_compute_pvalue.ipynb

# Validate the FPR control using normal reference patches
jupyter notebook ex2_validity_of_pvalue.ipynb
```

## How It Works

PADI treats the frozen encoder as a piecewise-affine function. It characterizes the sampling distribution of the test statistic by constructing a truncation region where the anomaly selection event remains unchanged. This reduces to a one-dimensional truncation problem along an affine line, allowing the selective $p$-value to be evaluated as the tail probability of a truncated Gaussian distribution.

![Figure 2: Geometric illustration of the truncation region](figures/figure%202.png)
