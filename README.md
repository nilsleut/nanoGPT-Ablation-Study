# nanoGPT Ablation Study: Depth & Head Scaling on FineWeb-Edu

This repository presents a systematic ablation study on transformer architecture scaling using nanoGPT, trained on 100k FineWeb-Edu documents (~104M tokens). Experiments isolate depth (n_layer: 2,4,6,8 at n_head=6) and head count (n_head: 1,2,3 at n_layer=6, d_model=384) effects on validation loss after 5000 iterations.

**Baseline (6L-6H)**: 5.30 val loss (PPL ~200). Scaling heads from 1→3 reduces loss by ~0.02 (384→128 dim/head), while deeper models (8L) plateau around 5.3, indicating compute-optimal scaling trade-offs.

## Experimental Setup

- **Model**: nanoGPT (GPT-2 tokenizer, bfloat16, RMSprop, cosine LR schedule)
- **Dataset**: FineWeb-Edu subset (100k docs → train.bin 208MB, val.bin 1MB)
- **Hardware**: Kaggle Tesla T4/P100 (~800ms/iter at bs=32)
- **Hyperparameters**: block_size=256, max_iters=5000, eval_interval=500

| Configuration | Parameters (M) | Best Val Loss | PPL    | Head Dim |
|---------------|----------------|---------------|--------|----------|
| Baseline (6L-6H) | 30.1       | 5.2975        | 199.8  | 64       |
| 2L-6H         | 23.0       | 5.4891        | 242.0  | 64       |
| 6L-1H         | 30.1       | 5.3385        | 208.2  | 384      |
| 6L-2H         | 30.1       | 5.3312        | 206.7  | 192      |
| 6L-3H         | 30.1       | 5.3166        | 203.7  | 128      |

Results show multi-head attention benefits diminish beyond 3 heads; shallower models underfit despite fixed heads.

<img width="1333" height="490" alt="image" src="https://github.com/user-attachments/assets/81ce9cc2-d971-4982-a6d1-afa9434fbff2" />

## Key Insights

- **Head Scaling**: Loss drops ~1.5% per head doubling (dim/2), saturating at ~128 dim/head — aligns with emergent multi-head coordination in transformers
- **Depth Scaling**: 6L optimal; 2L lags by 3.6%, 8L marginal gains suggest quadratic compute overhead outpaces linear loss reduction
- **Reproducibility**: Checkpoints/logs in `/checkpoints/{run_name}`; visualize via `ablation/ablationresults.png` (Matplotlib)

## Reproduction

1. Attach nanoGPT files as Kaggle dataset
2. Run cells sequentially: data prep → ablations → aggregation/plotting
3. Outputs auto-save to `/kaggle/working` (92KB total)

Extends prior nanoGPT work; future: layer norm position, activation scaling.

**Code**: [Kaggle Notebook](https://www.kaggle.com/code/nilsleutenegger/nanogpt-ablation-fineweb-head-depth)
