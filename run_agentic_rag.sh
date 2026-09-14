#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python agentic_rag.py \
    --model_name "qwen-2.5-32b" \
    --num_rows 100 \
    --temperature 0.7 \
    --output_dir "agentic_rag_output" \
    --checker_model_name "gemma-2-9b" \
    --max_steps 12 \
    --refinement_passes 3 \
    --top_k 5 \
    --seed 42
