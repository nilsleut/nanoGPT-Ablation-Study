# nanoGPT-Ablation-Study
This repository presents a systematic ablation study on transformer architecture scaling using nanoGPT, trained on 100k FineWeb-Edu documents (~104M tokens). Experiments isolate depth (n_layer: 2,4,6,8 at n_head=6) and head count (n_head: 1,2,3 at n_layer=6, d_model=384) effects on validation loss after 5000 iterations.
