\# Benchmark Experimental Results



This directory contains the experimental results obtained from the MINLPLib and QPLIB benchmark instances, together with the Synthetic′ dataset, for the study \*\*Learning to Relax Non-Convex QCQP within a Spatial Branch-and-Bound Context\*\*.



The experiments investigate the selection of LP and semidefinite-based relaxations using classical machine learning models and graph neural networks (GNNs), considering both root-node relaxation comparisons and spatial branch-and-bound (B\&B) performance.



\## Directory Overview



| Directory | Description |

|---|---|

| `gnn\_B\&B\_synprime\_8016\_.../` | GNN classification and regression results for spatial B\&B experiments. |

| `gnn\_ROOT\_8\_designs\_UNBALANCED\_.../` | GNN classification and regression results for root-node relaxation selection. |

| `sklearn\_root\_synprime\_8016\_.../` | Classical machine learning results for root-node experiments. |

| `sklearn\_all\_runs\_synthetic\_prime\_8016\_.../` | Classical machine learning results for spatial B\&B experiments, including static and dynamic feature configurations. |



\## Experimental Configurations



The benchmark evaluation considers:



\- \*\*Datasets:\*\* MINLPLib, QPLIB, and Synthetic′.

\- \*\*Relaxation comparisons:\*\* LP versus SDP and LP versus SDP′, as applicable to each experiment.

\- \*\*Learning tasks:\*\* Classification and regression.

\- \*\*Classical machine learning models:\*\* XGBoost, Gradient Boosting, and Random Forest.

\- \*\*Graph neural networks:\*\* Tripartite graph representations of QCQP instances.

\- \*\*Training and evaluation settings:\*\* Combined benchmark training and testing, and transfer learning from Synthetic′ to the combined benchmark.

\- \*\*Feature configurations:\*\* Static dimension-independent features and early spatial B\&B information.



\## Results and Evaluation



Depending on the experiment, the output directories contain model performance summaries, predictions, classification diagnostics, regression metrics, and experiment-specific metadata.



The classification experiments evaluate relaxation-selection accuracy, while the regression experiments predict the relative difference between relaxation objective values.



The spatial B\&B experiments additionally investigate the use of relaxation selection within the branch-and-bound search process.



\## Reproducibility



The results should be interpreted alongside the experimental methodology and dataset definitions in the accompanying manuscript.



Individual experiment directories may contain saved model outputs, train/test split information, and intermediate diagnostic files. The directory names include experiment identifiers and timestamps to distinguish different runs.



The scripts used to generate these results are provided separately in the repository's `code/` directory.

