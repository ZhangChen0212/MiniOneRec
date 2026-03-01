#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf
accelerate launch --num_processes 1 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon/index \
    --plm_checkpoint Qwen/Qwen3-Embedding-4B
