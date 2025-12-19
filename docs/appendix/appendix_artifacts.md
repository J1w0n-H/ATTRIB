# Appendix Artifacts (Auto-generated)

## A. Summary Metrics (52-case evaluation slice)

| System output | Accuracy | Main P/R | Dep P/R | Notes |
|---|---:|---:|---:|---|
| Stage 1 (Individual) | 84.62% | 97.30% / 83.72% | 53.33% / 88.89% | Derived from `llm_inference_results_checkpoint_module1_bugtype_groups.json` |
| Stage 3 (Synthesis) | 65.38% | 93.10% / 62.79% | 30.43% / 77.78% | From `evaluation_results.json` |
| ARVO submodule-only baseline | 82.69% | 82.69% / 100.00% | 0.00% / 0.00% | Predict Dep iff `arvo.submodule_bug==1` |

## B. Confusion Matrices (GT = heuristic labels)

### Stage 1
```
                    Predicted Main    Predicted Dep    Total
Actual Main              36                7           43
Actual Dependency          1                8             9
Total                    37               15           52
```

### Stage 3
```
                    Predicted Main    Predicted Dep    Total
Actual Main              27               16           43
Actual Dependency          2                7             9
Total                    29               23           52
```

## C. Stage 3 vs Stage 1 Disagreement Summary

- Total disagreements: **14**
- Stage 3 improved over Stage 1 (Stage1 wrong → Stage3 correct): **2**
- Stage 3 degraded vs Stage 1 (Stage1 correct → Stage3 wrong): **12**

See: `stage1_vs_stage3_disagreements.csv` for full rows.

