# GitHub Release Package (Paper + Code + Results)

This folder is a **self-contained, GitHub-ready** package for the paper, the core pipeline scripts, and the artifacts needed to reproduce the **paper metrics**.

## Folder Structure

- `paper/`
  - `paper_final.md`: final paper
- `code/`
  - `01_extract_from_db.py`
  - `02_build_ground_truth.py`
  - `03_llm_inference_modules.py`
  - `04_evaluate_llm_inference.py`
- `docs/`
  - `01_EXTRACT_SUMMARY.md`, `02_BUILD_GROUND_TRUTH_SUMMARY.md`, `03_LLM_INFERENCE_SUMMARY.md`, `04_EVALUATE_SUMMARY.md`
  - `appendix/`: appendix artifacts used during analysis
- `artifacts/`
  - `llm_inference_results.json`: full 517-case inference output (used by the paper)
  - `evaluation_results_uuv517.json`: stored paper evaluation results (reference output)
  - `ground_truth_uuv517.json`: **GT subset** for 517 cases (small; derived from evaluation output)
  - `llm_inference_results_partial.json`: 52-case pilot inference output
  - `evaluation_results_uuv52_regen.json`: stored 52-case evaluation results (reference output)
  - `ground_truth_uuv52.json`: GT subset for 52 cases
  - Stage checkpoints for ablation: `llm_inference_results_checkpoint_module1_bugtype_groups.json`, `llm_inference_results_checkpoint_module2_subgroups.json`

## Prerequisites

- **Python 3.9+**
- Optional (only for baseline-from-DB features): `arvo.db`

### `arvo.db` location assumption

If you want to run scripts that query ARVO DB (e.g., extraction, some baseline checks), place `arvo.db` at:

- `./arvo.db` (same directory as this `README.md`)

You can also override with:

```bash
export ARVO_DB_PATH=/absolute/path/to/arvo.db
```

## Quickstart: Reproduce Paper Metrics (517-case + 52-case)

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./reproduce_paper_metrics.sh
```

Outputs:
- `artifacts/evaluation_recomputed_uuv517.json`
- `artifacts/evaluation_recomputed_uuv52.json`

## Notes on Ground Truth

The full `ground_truth.json` in the original workspace is ~520MB and is **not shipped** in this GitHub package.
Instead, this package includes **case-subset GT files** (`ground_truth_uuv517.json`, `ground_truth_uuv52.json`) that are sufficient to reproduce the reported paper metrics on the evaluated slices.

## Running the full LLM inference pipeline

`code/03_llm_inference_modules.py` requires an LLM API key and additional runtime setup (network access, provider credentials). The shipped artifacts already contain the inference outputs used for the paper.



