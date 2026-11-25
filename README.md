# Scientific-Knowledge-Graph

**Citation intelligence & provenance**

[![Validation](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/research.yml/badge.svg?branch=main&event=push)](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/research.yml)

Scientific citation-network analysis and graph-enhanced publication classification using Cora, with optional OpenAlex ingestion.

## Dashboard

[**Open interactive dashboard**](https://rohanst14.github.io/Scientific-Knowledge-Graph/) · [Evaluation protocol](docs/research-protocol.md)

Explore directed citation neighborhoods, publication categories, PageRank rankings, citation provenance and held-out classification metrics. Search by publication ID or select a node to navigate its neighborhood.

## Benchmark results

Cora: 2,708 publications, 5,429 directed citation edges, 1,895 training nodes and 813 held-out nodes.

| Method | Accuracy | Macro-F1 |
|---|---:|---:|
| Text-only logistic regression | 78.35% | 76.62% |
| Graph-neighborhood features + logistic regression | 88.93% | 88.64% |

[Measured report](docs/results/citations.json) · [Completed benchmark workflow](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/runs/45698062515). Results reflect a single stratified transductive split (seed 42), not an inductive or multi-seed benchmark.

## Installation and execution

```bash
git clone https://github.com/rohanst14/Scientific-Knowledge-Graph.git
cd Scientific-Knowledge-Graph
python -m venv .venv
# Activate the environment for your operating system.
pip install -e . -r requirements-research.txt
python experiments/cora_benchmark.py
python -m unittest discover -s tests -v
```

## Model selection

The benchmark also evaluates normalized multi-hop graph features (depth 0–4) and four regularization settings. Selection uses macro-F1 on a development subset of training nodes; the selected model is refitted on training nodes before evaluation on the unchanged held-out split. The measured JSON report includes the chosen configuration.

## Dataset and methodology

The default benchmark uses the public LINQS Cora citation dataset (2,708 publications and 5,429 input citation rows). OpenAlex ingestion remains optional and requires an API key or local snapshot. Compare text-only logistic regression with graph-neighborhood features on a stratified held-out node split; this is transductive evaluation.

Metrics: Held-out accuracy, macro-F1, citation-edge integrity, graph connectivity and PageRank.

See [evaluation protocol](docs/research-protocol.md) for assumptions and limitations. The experiment writes `results/citations.json`. Successful benchmark jobs publish measured JSON reports to the repository; the dashboard loads the published report automatically when available. 

## Repository organization

| Path | Purpose |
|---|---|
| src/ | Core algorithms and model interfaces |
| experiments/ | Reproducible experiment entry points |
| tests/ | Automated regression tests |
| docs/ | Dashboard and methodology |
| requirements-research.txt | Experiment dependencies |
| .github/workflows/ | Continuous validation |

Use [GitHub Actions](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions) to inspect actual test outcomes.
