# Learning to Relax Non-Convex QCQP within a Spatial Branch-and-Bound Context

Research code and computational results accompanying the study **Learning to Relax Non-Convex QCQP within a Spatial Branch-and-Bound Context**.

The study investigates how machine learning can select between a computationally cheaper linear-programming (LP) relaxation and stronger semidefinite-based relaxations (SDP and SDP′) for nonconvex quadratically constrained quadratic programs (QCQPs). It includes synthetic-instance experiments, benchmark experiments using MINLPLib and QPLIB, root-node comparisons, and spatial branch-and-bound (B&B) experiments.

## Repository guide

| Location | Contents |
|---|---|
| [`code/`](code/) | Research scripts and notebooks for instance generation, feature extraction, relaxation solving, classical machine learning, GNNs, and B&B experiments. |
| [`ExperimentResultsRootNode/`](ExperimentResultsRootNode/) | Results from the original root-node experiments on synthetic instance families, including classical classifiers/regressors and GNNs. |
| [`ExperimentResultsBenchmark/`](ExperimentResultsBenchmark/) | Results for the revised benchmark evaluation, with combined MINLPLib/QPLIB instances and Synthetic′-to-benchmark transfer settings; includes root-node and B&B experiment outputs. |

The directories above may contain their own README files with more specific notes.

## Understanding the root-node results

The `ExperimentResultsRootNode/` directory organizes outputs by **learning task**, **feature configuration**, and **experimental dataset**. Examples of folder names include:

| Folder name or prefix | How to read it |
|---|---|
| `clf_outputs_...` | Classical machine-learning **classification** outputs. |
| `reg_outputs_...` | Classical machine-learning **regression** outputs. |
| `gnn_clf_outputs...` | GNN classification outputs. |
| `gnn_reg_outputs...` | GNN regression outputs. |
| `...synth...` | Experiments involving the original synthetic instance dataset. |
| `...synthprime...` or `...prime` | Experiments involving the modified **Synthetic′** dataset. |
| `...txt_features...` | Runs using saved, tabular feature representations; exact feature sets should be checked in the corresponding experiment configuration. |
| `...missings...` | A separately named experimental configuration; check its run files for the precise missing-feature treatment. |

Many root-node output folders have names encoding the relaxation comparison, the training family or families, and the test family, for example:

`DI_binary__lpvsdp__train[5_1]__test[holdout]`

The components indicate:

- **`DI_binary`**: a binary classification experiment using dimension-independent (DI) features.
- **`lpvsdp` / `lpvsdp_prime`**: the relaxation pair being compared (LP against SDP or SDP′).
- **`train[...]`**: the instance family or combination of families used for training, identified by the experiment's family codes.
- **`test[...]`**: the test family or a held-out test partition. `holdout` indicates an evaluation subset kept separate from training.

Family codes such as `5_1`, `10_2`, and `10_2+5_2` are **experiment identifiers**, not self-explanatory mathematical definitions. Their exact instance-generation settings are defined in the relevant generator and experiment scripts; the code should be consulted rather than inferring dimension or structure from the code alone.

Individual root-node run folders may contain files such as:

| File prefix | Interpretation |
|---|---|
| `results_clf_...csv` / corresponding regression summaries | Performance summary for that run. |
| `preds_xgb_...csv`, `preds_rf_...csv`, `preds_gbc_...csv` | Per-instance model predictions for XGBoost, Random Forest, and Gradient Boosting, respectively. Other model-specific prefixes may occur. |
| `cm_xgb_...csv`, `cm_rf_...csv`, `cm_gbc_...csv` | Confusion matrices for the indicated classifier. |
| `cm_...png` | Visualized confusion matrices. |

## Understanding the benchmark results

`ExperimentResultsBenchmark/` contains the later experiments on the **combined MINLPLib and QPLIB benchmark** and experiments trained using **Synthetic′** and tested on the combined benchmark.

The folder names encode the model family and whether the experiment concerns a root-node decision or a spatial B&B setting:

| Folder name or prefix | Interpretation |
|---|---|
| `gnn_B&B_synprime_8016_...` | GNN experiments in the B&B setting, associated with the Synthetic′ experimental dataset. |
| `gnn_ROOT_8_designs_UNBALANCED_...` | Root-node GNN experiments covering several combinations of task, relaxation pair, and training/testing setting. |
| `sklearn_root_synprime_8016_...` | Root-node classical machine-learning runs. |
| `sklearn_all_runs_synthetic_prime_8016_...` | Classical machine-learning runs for the broader benchmark/B&B experimental configurations. |

The date/time suffixes distinguish saved runs; they do not represent separate model types.

A run folder such as `run_01__combined_to_combined__classification` expresses its **training/evaluation setting** and **learning task**:

- `combined_to_combined`: train and evaluate on disjoint portions of the combined benchmark dataset.
- `synthetic_to_combined`: train on Synthetic′ and evaluate on the combined benchmark.
- `classification`: predict the favored relaxation label.
- `regression`: predict a relative-difference target between relaxation objective values.

Some run names also specify the LP-versus-SDP or LP-versus-SDP′ comparison; consult the run's experiment manifest when the folder name is abbreviated.

Common files in benchmark run directories include:

| File | Purpose |
|---|---|
| `all_gnn_results.csv`, `results_gnn.csv`, `thesis_style_gnn_results.csv` | Aggregate or run-level model-performance summaries. |
| `preds_test.csv` | Test-instance identifiers, true targets and model predictions; actual columns depend on the task. |
| `cm.csv` / `cm.png` | Classification confusion matrix in numeric/visual form. |
| `instance_split.csv` | Instances assigned to training and testing for a particular run. |
| `combined_train_test_split_manifest.csv` | Recorded benchmark-level train/test membership, where provided. |
| `experiment_manifest.csv` | Identifiers and settings of the saved experiments. |
| `modeling_removed_combined.csv`, `modeling_removed_synthetic.csv` | Records of instances excluded from a particular modeling workflow. |
| `epoch_log.csv` | Per-epoch training-loss history for neural-network runs. |
| `model.pt` | Saved trained PyTorch model weights. |

Not every file appears in every run, and metric availability varies between classification and regression.

## Targets and evaluation

In each LP-versus-alternative comparison, the class labels represent which relaxation is favored under the study's tolerance-based bound-quality criterion:

- **Class 1:** LP is favored (including sufficiently close comparisons for which its lower computational cost motivates selecting LP).
- **Class 0:** The SDP or SDP′ alternative is favored.

The regression task predicts the relative difference, denoted by **delta (δ)** in the manuscript. Refer to the paper for its exact mathematical definition and tie tolerance.

Results should be interpreted with their **relaxation comparison, data split, instance family, and feature representation**; accuracy values from different run folders are not necessarily directly comparable without these details.

## Reproducing and using the outputs

1. Read the paper's computational setting and relevant folder README before selecting an experiment.
2. Use the run names and manifest files to identify the dataset, relaxation comparison, task, and train/test protocol.
3. Use `preds_test.csv` or model-specific `preds_*.csv` for per-instance evaluation; use result-summary CSV files for reported performance metrics.
4. Use the code in `code/` for implementation details. Some notebooks require local input data, intermediate files, and licensed optimization solvers; a single end-to-end executable pipeline is not implied.

**Scope note:** This README explains naming conventions observed in the research output folders; it does not certify that every intermediate run appears in the manuscript or that all directories are immediately executable. The manuscript and the explicitly identified final experiment outputs take precedence when interpreting reported results.
