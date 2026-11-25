# Experimental protocol

## 1. Objective and data

- Task: seven-class publication subject classification on the LINQS Cora citation network.
- Data: 2,708 nodes, 5,429 directed citation edges and 1,433 binary word-presence features per node.
- Source: [LINQS Cora](https://linqs.org/datasets/) via the [PyGCN mirror](https://github.com/tkipf/pygcn/tree/master/data/cora).
- Integrity: record SHA-256 of the content bytes, a newline separator and citation bytes; validate node identifiers, edge endpoints and counts.
- Citation rows are ordered cited ID, citing ID; retain citing-to-cited directed edges for exploration. For classification, symmetrize adjacency, add self-loops and row-normalize to obtain P.
- Original article titles and abstracts are not present in these data. OpenAlex is a separate optional ingestion path and is not used in the benchmark.

## 2. Partitioning and leakage control

- Primary partition: stratified 70/30 split with seed 42; 1,895 training nodes and 813 test nodes.
- Model selection: split training into 1,516 development and 379 validation nodes, using seed 42 and stratification.
- Use the same primary partitions for all four classifiers. Fit candidate models on development labels; choose configurations using validation scores only, then refit on the full training set.
- Transductive setting: the full graph topology and unlabeled feature matrix are available to every model; test labels are excluded from model fitting, configuration selection and label propagation.
- The primary benchmark was inspected in earlier development. Later comparisons are exploratory, not fresh independent confirmation.

## 3. Classifiers and selection

- Text-only baseline: multinomial logistic regression on original binary feature matrix X.
- One-hop baseline: logistic regression on concatenated X and PX.
- Multi-hop: L2-normalize X by row, propagate through P up to depth k, normalize each propagated block, concatenate the blocks.
- Multi-hop search: k = 0–4 and C ∈ {0.1, 1, 10, 100}; 20 configurations selected by validation macro-F1.
- Kernel/diffusion: combine 85% cosine similarity of normalized multi-hop feature vectors with 15% cosine similarity of normalized four-step transition-matrix rows.
- Kernel search: RBF exp(-gamma × squared distance), gamma ∈ {0.5, 2}, SVM C ∈ {1, 10, 100}, plus the selected logistic model. Estimate SVM probabilities internally on training examples.
- Residual correction strengths {0, 1, 3}; clamped probability smoothing weights {0, 0.5, 0.9}; 30 propagation steps. Construct residuals and clamped labels exclusively from the supplied training subset.
- Kernel/diffusion search: 63 configurations; select by validation accuracy with macro-F1 as tie-breaker. Refit on full training partition.

## 4. Evaluation

- Hard predictions: accuracy, macro precision, balanced accuracy (macro recall), macro-F1, support-weighted F1, Matthews correlation coefficient and Cohen's kappa.
- Ranking: macro one-vs-rest ROC-AUC and average precision, with equal class weights and probability columns ordered by sorted class labels.
- Probability quality: natural-log loss; multiclass Brier score (mean sum of squared probability errors, range 0–2); top-label expected calibration error (15 equal-width confidence bins, weighted by bin population).
- Prefer higher classification and ranking metrics, lower probability errors. Undefined class precision is zero.
- Record per-class precision, recall, F1 and support; confusion matrices (true rows, predicted columns), reliability bins and paired correctness counts. Report paired changes descriptively; connected nodes are dependent.
- Keep the 95% accuracy goal separate from measured performance; it is not a CI pass criterion.

## 5. Sensitivity and limitations

- Repeat stratified splits with seeds 17, 29 and 73 for text-only, one-hop and fixed multi-hop configurations.
- Report mean and sample standard deviation of accuracy and macro-F1; kernel/diffusion is evaluated only on the primary split.
- These partitions overlap with each other and with earlier development data. Their variability is descriptive, not an independent confidence interval or external validation.
- Cora is a historical dataset; results do not establish generalization to other scientific fields, inductive graphs or state-of-the-art superiority.

## 6. Reproducibility

- Install the dependencies in `requirements-research.txt`.
- Execute `python experiments/cora_benchmark.py`.
- Validate output with `python tests/check_report.py results/citations.json` and run the repository unit tests.
- The browser dashboard reads the published JSON; it does not train classifiers. Node colors represent dataset annotations, not predictions.
- [Experimental report](experimental-report.md) · [Measured JSON](results/citations.json) · [Dashboard](https://rohanst14.github.io/Scientific-Knowledge-Graph/).
