# Cora experimental report

- Dataset: LINQS Cora (PyGCN mirror).
- SHA-256: `9bd61b0211155dfda9e1280e00fef9fbcaedb5befee6042e121d6473fa2af80f`.
- Report schema: 3; seed: 42; training: 1895; test: 813.
- [Completed benchmark workflow](https://github.com/rohanst14/Scientific-Knowledge-Graph/actions/runs/45700542187) · [Raw results](results/citations.json) · [Experimental protocol](research-protocol.md).

## Held-out performance

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

## Selected configurations

- Multi-hop: depth 4, C = 1; validation macro-F1 0.8653; 20 candidates.
- Kernel/diffusion: kernel_svm, C = 1, gamma = 0.5, correction = 1, smoothing = 0; validation accuracy 88.39%, validation macro-F1 0.8687; 63 candidates.

## Class-level diagnostics

| Category | Multi-hop precision | Multi-hop recall | Multi-hop F1 | Kernel/diffusion F1 | Support |
|---|---:|---:|---:|---:|---:|
| Case_Based | 0.8646 | 0.9326 | 0.8973 | 0.8830 | 89 |
| Genetic_Algorithms | 0.9764 | 0.9841 | 0.9802 | 0.9764 | 126 |
| Neural_Networks | 0.9084 | 0.9268 | 0.9175 | 0.9039 | 246 |
| Probabilistic_Methods | 0.9573 | 0.8750 | 0.9143 | 0.9106 | 128 |
| Reinforcement_Learning | 0.8806 | 0.9077 | 0.8939 | 0.8824 | 65 |
| Rule_Learning | 0.9574 | 0.8333 | 0.8911 | 0.8889 | 54 |
| Theory | 0.8056 | 0.8286 | 0.8169 | 0.8131 | 105 |

## Paired outcomes

| Reference → candidate | Both correct | Reference only | Candidate only | Both wrong | Accuracy change (pp) |
|---|---:|---:|---:|---:|---:|
| Text only → multi-hop | 604 | 33 | 134 | 42 | 12.42 |
| One-hop → multi-hop | 701 | 22 | 37 | 53 | 1.85 |
| Multi-hop → kernel/diffusion | 724 | 14 | 7 | 68 | -0.86 |

## Split sensitivity

| Seed | Text accuracy | One-hop accuracy | Multi-hop accuracy | Multi-hop macro-F1 |
|---|---:|---:|---:|---:|
| 17 | 76.75% | 86.22% | 88.93% | 87.11% |
| 29 | 77.61% | 86.59% | 89.18% | 88.09% |
| 73 | 77.86% | 88.31% | 89.30% | 88.44% |

- Overlapping splits provide descriptive sensitivity checks, not independent validation.
- Kernel/diffusion was evaluated only on seed 42.
- The benchmark has been reused during development; no independent external dataset or state-of-the-art claim is established.
- Confusion matrices, reliability bins and the kernel/diffusion validation candidate grid are available in the raw results JSON.
