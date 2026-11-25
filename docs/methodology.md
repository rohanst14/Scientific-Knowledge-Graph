# Controlled-text extraction example

This optional rule-based example is separate from the [Cora experimental protocol](research-protocol.md).

- Inputs: a mapping of source IDs to text containing simple affirmative sentences.
- Supported verbs: increases, decreases, causes, inhibits, predicts.
- Outputs: directed subject–relation–object records with source IDs; distinct simple paths bounded by edge count.
- Basic negation markers are rejected conservatively; complex negation, modality, coreference, syntax and entity disambiguation are not supported.
- A relation is a textual assertion, not verified causality. Regression tests exercise the interface; no annotated extraction benchmark has been performed.

Run `python examples/demo.py`.
