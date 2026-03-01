accelerate launch --num_processes 1 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon/index \
    --plm_checkpoint Qwen/Qwen3-Embedding-4B
