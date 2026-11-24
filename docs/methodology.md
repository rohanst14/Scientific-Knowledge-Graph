# Methodology

## Objective
Convert controlled scientific sentences into directed, source-tagged triples and explore bounded paths between entities.

## Algorithm
Split sentences → match supported relation verbs → normalize entities → build directed adjacency → traverse without cycles.

## Interface
extract(documents); paths(triples, start, target, max_depth=4)

Inputs: A mapping of source identifiers to text; graph endpoints and depth bound.

Outputs: A list of subject–relation–object–source records and a list of simple node paths.

## Evaluation protocol
Measure entity and relation precision, recall and F1 against manually annotated sentences; test duplicate edges, negation and multi-hop paths.

Report test corpus provenance, exact revision, configuration, failures and uncertainty. The included regression tests are not a benchmark.

## Threats to validity
Rule-based extraction does not establish causality; complex syntax, negation and coreference are unsupported.
