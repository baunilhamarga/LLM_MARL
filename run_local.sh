#!/usr/bin/env bash
set -euo pipefail

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/usr/users/xai/gama_hei/projects/llm_dce/models/huggingface}"
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not available"; print(f"Using GPU: {torch.cuda.get_device_name(0)}")'

python dragonExp.py \
    --provider local \
    --model Llama-3.1-8B-Instruct \
    --model_path "$MODEL_PATH" \
    --model_cache_dir "$MODEL_CACHE_DIR" \
    --local_dtype float16 \
    --max_completion_tokens 256 \
    --exp_name llama-local-smoke \
    --preset default \
    --allow_comm \
    --max_step 3 \
    --temperature 0 \
    --seed 0 \
    "$@"

