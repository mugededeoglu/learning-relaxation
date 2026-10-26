# Benchmark and Spatial Branch-and-Bound Experiments

This directory contains the implementation and experimental notebooks developed for the paper **Learning to Relax Non-Convex QCQP within a Spatial Branch-and-Bound Context**.

The experiments investigate LP, SDP, and SDP′ relaxation selection for nonconvex quadratically constrained quadratic programs (QCQPs), using synthetic instances and the MINLPLib and QPLIB benchmark libraries.

## Code Overview

### 1. Dataset Generation

| File | Description |
|---|---|
| `QPLIB_BB_dataset_generation.ipynb` | QPLIB instance preparation and dataset generation for spatial B&B experiments. |
| `data_generator_MINIMAL_Cell1_Cell2_only.ipynb` | Synthetic QCQP instance generation. |

### 2. Relaxation Solvers and Spatial Branch-and-Bound

| File | Description |
|---|---|
| `qcqp_solversdg.py` | QCQP relaxation solver implementation. |
| `branchandboundreviewsynthetic.ipynb` | Spatial B&B experiments on synthetic instances. |
| `reviewbbgnn.ipynb` | GNN experiments in the spatial B&B setting. |
| `method1trainingreview.ipynb` | Training and calibration experiments associated with the B&B methodology. |

### 3. Root-Node Experiments and Machine Learning

| File | Description |
|---|---|
| `reviewrootnode.ipynb` | Root-node relaxation comparisons and ML experiments. |
| `featureextractionreview.ipynb` | Extraction of dimension-independent features. |
| `combined_static_features_fixed.ipynb` | Static features for the combined MINLPLib and QPLIB benchmarks. |
| `synthetic_static_features_fixed.py` | Static feature extraction for synthetic instances. |
| `gnn_overhead_timing.ipynb` | Measurement of GNN-related computational overhead. |

### 4. Additional Analyses

| File | Description |
|---|---|
| `ablationstudy.ipynb` | Feature ablation experiments. |
| `analysis3.ipynb` | Additional experimental analysis. |
| `analysis4.ipynb` | Additional experimental analysis. |
| `analysis5.ipynb` | Additional experimental analysis. |

## Experimental Settings

The computational study considers two principal settings:

- **Root-node relaxation selection:** Comparing the strength of LP, SDP, and SDP′ relaxations at the root node.
- **Spatial branch-and-bound:** Evaluating relaxation selection within a spatial B&B framework.

Classification and regression models are trained on synthetic or benchmark instances. The graph neural network uses a tripartite representation of the QCQP.

## Supplementary Results

Detailed classification and regression results, baseline comparisons, class-label distributions, confusion matrices, and additional evaluation metrics are provided separately in the repository's `results/` directory.

## Reproducibility

The notebooks represent individual research experiments and may require external solver installations, local datasets, and intermediate files. They should not be interpreted as a single executable pipeline.
