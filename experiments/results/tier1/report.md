# TabEBM Reproduction Report

**Results dir**: `experiments/results/tier1`  
**Valid rows**: 90  
**Datasets**: biodeg, stock  
**Methods**: baseline, smote, tabebm  
**Classifiers**: knn, rf, tabpfn


## Table 1: Balanced Accuracy (%) per Dataset

| Method | biodeg | stock | **Avg** |
|---|---|---|---|
| **baseline** | 78.49 ± 2.75 | 91.34 ± 3.37 | **84.91** |
| **smote** | 76.46 ± 3.73 | 92.34 ± 2.50 | **84.40** |
| **tabebm** | 76.86 ± 3.11 | 91.85 ± 2.34 | **84.35** |

## Average Rank (lower = better)

| Method | Rank ± std |
|---|---|
| **baseline** | 1.633 ± 0.776 |
| **smote** | 2.100 ± 0.814 |
| **tabebm** | 2.267 ± 0.728 |

## ADTM (higher = better, affine-normalized)

| Method | ADTM |
|---|---|
| **baseline** | 0.6910 |
| **smote** | 0.5017 |
| **tabebm** | 0.3389 |

## Coverage

| Dataset | OK | Skipped | Failed |
|---|---|---|---|
| biodeg | 45 | 0 | 0 |
| stock | 45 | 0 | 0 |