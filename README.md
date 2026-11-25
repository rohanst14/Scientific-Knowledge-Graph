# Scientific-Knowledge-Graph

[![Validation](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/research.yml)

Citation-network analysis and publication subject classification using multi-hop graph propagation and multinomial logistic regression. Directed citation relationships are preserved for network exploration; an undirected adjacency matrix is used for feature propagation.

[**Interactive dashboard**](https://rohanst14.github.io/Scientific-Knowledge-Graph/) · [Experimental protocol](docs/research-protocol.md) · [Experimental report](docs/experimental-report.md) · [Results JSON](docs/results/citations.json)

## Dataset

[Cora (LINQS)](https://linqs.org/datasets/) contains 2,708 publications, 5,429 directed citation edges, seven subject categories and 1,433 binary word-presence features per publication. The experiment uses the [PyGCN data mirror](https://github.com/tkipf/pygcn/tree/master/data/cora). Publication identifiers and citation directions are retained; original paper titles and abstracts are not supplied by these files.

## Method

Let X denote the binary feature matrix and P the row-normalized undirected adjacency matrix with self-loops. Three multinomial logistic regression classifiers are compared:

- **Text only:** original word-presence features X.
- **One-hop:** concatenate X and PX.
- **Multi-hop:** L2-normalize each row of X, propagate with P up to depth k, L2-normalize each resulting feature block, and concatenate the blocks. Select k from 0–4 and regularization C from {0.1, 1, 10, 100} using validation macro-F1.

The seed-42 stratified split assigns 1,895 nodes to training and 813 to testing. For multi-hop model selection, 379 training nodes serve as validation and 1,516 as development data. The selected model (k = 4, C = 1) is refitted on all 1,895 training nodes. The full graph and unlabeled features are available during training (transductive evaluation); test labels are excluded from fitting and model selection. This is fixed graph feature propagation, not a trained graph neural network.

## Kernel and label-diffusion comparison

The additional classifier compares regularized logistic regression with nonlinear precomputed-kernel SVMs using normalized multi-hop text similarity and four-step graph-diffusion coordinates. Training-label residual correction and clamped probability smoothing are selected together with the classifier from 63 configurations on the development validation partition, by accuracy with macro-F1 as tie-breaker. Only training labels are supplied to the diffusion algorithm. The selected configuration is refitted on all training nodes. The completed comparison is included below. Because the seed-42 test scores have already been inspected in earlier development, the comparison is exploratory on a reused benchmark rather than fresh independent confirmation.

## Results

All metrics below are calculated on the same 813 test nodes. Scores are reported on their native scales; bold indicates the best value among the four models.

| Metric | Text only | One-hop | Multi-hop | Kernel/diffusion | Preferred |
|---|---:|---:|---:|---:|:---:|
| Accuracy | 0.7835 | 0.8893 | **0.9077** | 0.8991 | ↑ |
| Macro precision | 0.7960 | 0.8952 | **0.9072** | 0.8981 | ↑ |
| Balanced accuracy | 0.7464 | 0.8809 | **0.8983** | 0.8938 | ↑ |
| Macro-F1 | 0.7662 | 0.8864 | **0.9016** | 0.8940 | ↑ |
| Weighted-F1 | 0.7818 | 0.8886 | **0.9079** | 0.8994 | ↑ |
| MCC | 0.7342 | 0.8649 | **0.8876** | 0.8774 | ↑ |
| Cohen's kappa | 0.7328 | 0.8645 | **0.8874** | 0.8771 | ↑ |
| Macro ROC-AUC (OvR) | 0.9479 | 0.9837 | **0.9890** | 0.9871 | ↑ |
| Macro average precision (OvR) | 0.8117 | 0.9383 | **0.9548** | 0.9460 | ↑ |
| Log loss | 0.7408 | 0.3843 | **0.3299** | 0.3321 | ↓ |
| Multiclass Brier score | 0.3341 | 0.1795 | **0.1516** | 0.1555 | ↓ |
| Expected calibration error | 0.0477 | 0.0375 | 0.0660 | **0.0300** | ↓ |

Multi-hop logistic regression has the highest measured accuracy (90.77%) and macro-F1 (90.16%). Kernel/diffusion achieves 89.91% accuracy (-0.86 percentage points relative to multi-hop) but lower top-label calibration error (0.0300 versus 0.0660). ROC-AUC measures ranking, not classification accuracy.

### Split sensitivity

The fixed multi-hop configuration was also evaluated on stratified splits with seeds 17, 29 and 73; kernel/diffusion was evaluated only on seed 42. Values are mean ± sample standard deviation (percentage points for SD).

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Text only | 77.41% ± 0.58% | 74.77% ± 0.34% |
| One-hop | 87.04% ± 1.12% | 85.33% ± 1.40% |
| Multi-hop | 89.13% ± 0.19% | 87.88% ± 0.69% |

These partitions overlap with each other and with the original development split; the sensitivity results are descriptive, not independent external validation. Comparisons establish improvement over the implemented baselines, not state-of-the-art performance.

The [benchmark report](docs/results/citations.json) contains class-level metrics, confusion matrices, reliability bins, paired prediction counts, split-level results and the dataset checksum. [Completed experiment](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/runs/45700542187).

## Run locally

```bash
git clone https://github.com/rohanst14/Scientific-Knowledge-Graph.git
cd Scientific-Knowledge-Graph
python3.11 -m venv .venv
# Activate the virtual environment for your OS.
pip install -e . -r requirements-research.txt
python experiments/cora_benchmark.py
python -m unittest discover -s tests -v
python tests/check_report.py results/citations.json
```

The experiment writes `results/citations.json`. Import this file in the browser dashboard to inspect a local run.

## Repository structure

| Path | Implementation |
|---|---|
| [src/scientific_knowledge_graph/](src/scientific_knowledge_graph/) | Citation ingestion, graph queries and evaluation metrics |
| [experiments/cora_benchmark.py](experiments/cora_benchmark.py) | Feature propagation, model selection and benchmark execution |
| [tests/](tests/) | Unit tests, report integrity and browser interaction tests |
| [docs/](docs/) | Citation explorer, performance diagnostics and published report |

### Optional citation ingestion

Analyze a local OpenAlex JSONL snapshot with `python experiments/citation_study.py --input data/openalex.jsonl`. For network retrieval, set `OPENALEX_API_KEY` and add `--download --query "process safety" --limit 300`. The resulting `results/openalex-citations.json` is a separate network-analysis report, not a Cora classification benchmark. The public dashboard imports Cora reports only.

The [controlled-text extraction example](examples/demo.py) is a separate rule-based demonstration, documented in [methodology.md](docs/methodology.md); it is not part of the measured Cora experiment.
