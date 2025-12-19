#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 code/04_evaluate_llm_inference.py \
  --inference-file artifacts/llm_inference_results.json \
  --gt-file artifacts/ground_truth_uuv517.json \
  --output artifacts/evaluation_recomputed_uuv517.json

python3 code/04_evaluate_llm_inference.py \
  --inference-file artifacts/llm_inference_results_partial.json \
  --gt-file artifacts/ground_truth_uuv52.json \
  --output artifacts/evaluation_recomputed_uuv52.json

echo "Wrote:"
echo "  artifacts/evaluation_recomputed_uuv517.json"
echo "  artifacts/evaluation_recomputed_uuv52.json"


