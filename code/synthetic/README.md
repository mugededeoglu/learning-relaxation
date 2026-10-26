
# Synthetic QCQP Experiments

This directory contains the Python scripts and Jupyter notebooks used for the original synthetic QCQP experiments in *Learning to Relax Non-Convex QCQP within a Spatial Branch-and-Bound Context*.

The experiments investigate the relative performance of LP, SDP, and SDP′ relaxations on synthetically generated nonconvex quadratically constrained quadratic programs (QCQPs), using supervised classification and regression models.

## Code Overview

| File | Description |
|---|---|
| `data_generatorforsyntheticgnn.ipynb` | Generation of synthetic QCQP instances. |
| `feature_extractionsetupsall.ipynb` | Extraction of structural and numerical features from QCQP instances. |
| `qcqp_parser.py` | Parsing and preparation of QCQP problem data. |
| `qcqp_solverfusion.py` | Implementation of QCQP relaxation formulations and solver routines. |
| `classificationminlplibsyntheticlastexperiment.ipynb` | Classification experiments involving synthetic training data and MINLPLib instances. |
| `regressionforsynthanminlpliblastexperiment.ipynb` | Regression experiments involving synthetic training data and MINLPLib instances. |
| `gnnclassifier.ipynb` | Graph neural network model for relaxation classification. |
| `gnnregressor.ipynb` | Graph neural network model for relaxation regression. |

## Experimental Scope

The experiments cover:

- Synthetic QCQP instance generation.
- Dimension-independent feature extraction.
- LP, SDP, and SDP′ relaxation comparisons.
- Supervised classification and regression.
- Graph neural network models using QCQP graph representations.
- Evaluation involving synthetic and MINLPLib instances.

## Reproducibility

These files represent the original experimental implementation. Some notebooks may require external optimization solvers, local datasets, and intermediate files.

The revised experiments involving QPLIB and spatial branch-and-bound are organized separately in the repository.
