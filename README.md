# Scientific-Knowledge-Graph

**Citation intelligence & provenance**

![Validation](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/research.yml/badge.svg)

Scientific citation-network analysis and graph-enhanced publication classification using Cora, with optional OpenAlex ingestion.

## Dashboard

[**Open interactive dashboard**](https://rohanst14.github.io/Scientific-Knowledge-Graph/) · [Evaluation protocol](docs/research-protocol.md)

Explore published benchmark metrics and import additional experiment reports in JSON format.

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
