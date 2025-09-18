# Qwen3-8B EAGLE3 Draft Training (UltraChat 1k)

## Prepare dataset (UltraChat, 1k)

```bash
python3 /sgl-workspace/specforge/scripts/prepare_data.py \
  --dataset ultrachat \
  --output-path /sgl-workspace/specforge/cache/dataset \
  --sample-size 1000
```

## Launch training (single GPU)

```bash
export PYTHONPATH=/sgl-workspace/specforge:$PYTHONPATH
export TORCHINDUCTOR_CACHE_DIR=/sgl-workspace/specforge/cache/compiled_kernels

torchrun --standalone --nproc_per_node 1 \
  /sgl-workspace/specforge/scripts/train_eagle3_online.py \
  --target-model-path Qwen/Qwen3-8B \
  --draft-model-config /sgl-workspace/specforge/configs/qwen3-8b-eagle3.json \
  --train-data-path /sgl-workspace/specforge/cache/dataset/ultrachat_train.jsonl \
  --output-dir /sgl-workspace/specforge/outputs/Qwen3-8B-eagle3-ultra1k \
  --num-epochs 10 \
  --batch-size 1 \
  --learning-rate 1e-4 \
  --max-length 2048 \
  --chat-template qwen \
  --cache-dir /sgl-workspace/specforge/cache \
  --embedding-key model.embed_tokens.weight \
  --tp-size 1 \
  --ttt-length 7 \
  --attention-backend flex_attention
```

Notes:
- The draft model uses Qwen3-specific attention with q/k head-wise RMSNorm, RoPE, and flex attention as the default backend.
- The training script will auto-cache processed data and vocab mapping under `/sgl-workspace/specforge/cache`.
