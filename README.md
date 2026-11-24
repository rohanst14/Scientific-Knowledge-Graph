# Scientific-Knowledge-Graph

### Provenance-aware scientific relation extraction

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white) ![Tests](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/workflows/tests.yml/badge.svg) ![Stage](https://img.shields.io/badge/stage-research%20prototype-64748b)

Convert controlled scientific sentences into directed, source-tagged triples and explore bounded paths between entities.

## Design

Split sentences → match supported relation verbs → normalize entities → build directed adjacency → traverse without cycles.

## Quick start

```bash
git clone https://github.com/rohanst14/Scientific-Knowledge-Graph.git
cd Scientific-Knowledge-Graph
python -m venv .venv
# Activate .venv for your operating system.
python -m pip install -e .
python examples/demo.py
python -m unittest discover -s tests -v
```

## Usage

```python
from scientific_knowledge_graph import extract, paths
triples = extract({'paper-1': 'Heat increases pressure. Pressure causes alarm.'})
print(paths(triples, 'heat', 'alarm'))
```

## API contract

| Field | Description |
|---|---|
| Entry point | `extract(documents); paths(triples, start, target, max_depth=4)` |
| Inputs | A mapping of source identifiers to text; graph endpoints and depth bound. |
| Outputs | A list of subject–relation–object–source records and a list of simple node paths. |

## Repository map

| Path | Role |
|---|---|
| [src/scientific_knowledge_graph/](src/scientific_knowledge_graph/) | Python implementation |
| [examples/demo.py](examples/demo.py) | Reproducible synthetic demo |
| [tests/test_core.py](tests/test_core.py) | Regression checks |
| [docs/methodology.md](docs/methodology.md) | Evaluation design and limitations |
| [ROADMAP.md](ROADMAP.md) | Milestones and planned work |
| [CI workflow](.github/workflows/tests.yml) | Automated tests and demo |

## Evaluation and limitations

Measure entity and relation precision, recall and F1 against manually annotated sentences; test duplicate edges, negation and multi-hop paths.

Rule-based extraction does not establish causality; complex syntax, negation and coreference are unsupported. The current demonstration is synthetic; no real-world performance claims are made.

## Project status

AI-assisted research prototype. See [roadmap](ROADMAP.md) for planned extensions.
