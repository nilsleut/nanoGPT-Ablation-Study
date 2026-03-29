"""
train.py — GPT training loop with checkpointing and W&B logging (optional).

Usage:
    # Baseline run
    python train.py

    # Custom config (overrides defaults)
    python train.py --n_layer=4 --n_head=4 --run_name=ablation_4L4H

    # Resume from checkpoint
    python train.py --resume=checkpoints/baseline/ckpt.pt

Saves to checkpoints/{run_name}/ckpt.pt every eval_interval steps.
"""

import os
import math
import time
import json
import argparse
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GPT, GPTConfig


# ── Argument parsing ──────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument('--n_layer',     type=int,   default=6)
    p.add_argument('--n_head',      type=int,   default=6)
    p.add_argument('--n_embd',      type=int,   default=384)
    p.add_argument('--block_size',  type=int,   default=256)
    p.add_argument('--dropout',     type=float, default=0.1)
    # Training
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--max_iters',   type=int,   default=5000)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--min_lr',      type=float, default=3e-5)
    p.add_argument('--warmup_iters',type=int,   default=200)
    p.add_argument('--grad_clip',   type=float, default=1.0)
    p.add_argument('--weight_decay',type=float, default=0.1)
    # Eval / logging
    p.add_argument('--eval_interval',  type=int, default=250)
    p.add_argument('--eval_iters',     type=int, default=100)
    p.add_argument('--log_interval',   type=int, default=50)
    # I/O
    p.add_argument('--data_dir',    type=str,   default='data')
    p.add_argument('--out_dir',     type=str,   default='checkpoints')
    p.add_argument('--run_name',    type=str,   default='baseline')
    p.add_argument('--resume',      type=str,   default=None)
    p.add_argument('--wandb',       action='store_true')
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────

def get_batch(split, data_dir, block_size, batch_size, device):
    """Load a random batch from the memory-mapped binary file."""
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data  = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix    = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i   : i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, eval_iters, device):
    """Average loss over eval_iters random batches for train and val."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = get_batch(split, data_dir, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = float(np.mean(losses))
    model.train()
    return out


# ── Learning rate schedule ────────────────────────────────────

def get_lr(it, warmup_iters, max_iters, lr, min_lr):
    """
    Linear warmup + cosine decay.
    Same schedule as nanoGPT and most modern LM training.
    """
    if it < warmup_iters:
        return lr * it / warmup_iters
    if it > max_iters:
        return min_lr
    progress = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


# ── Main ──────────────────────────────────────────────────────

def main():
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.bfloat16 if (device == 'cuda' and
              torch.cuda.is_bf16_supported()) else torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    # Output directory
    ckpt_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load vocab size from meta
    meta_path = os.path.join(args.data_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    vocab_size = meta['vocab_size_padded']   # 50304

    # Model
    cfg = GPTConfig(
        block_size = args.block_size,
        vocab_size = vocab_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
        dropout    = args.dropout,
    )

    iter_num   = 0
    best_val   = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt      = torch.load(args.resume, map_location=device)
        cfg       = GPTConfig(**ckpt['config'])
        model     = GPT(cfg).to(device)
        model.load_state_dict(ckpt['model'])
        iter_num  = ckpt['iter_num']
        best_val  = ckpt['best_val']
    else:
        model = GPT(cfg).to(device)

    n_params = model.num_parameters()
    print(f"Model: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}d  "
          f"Parameters: {n_params/1e6:.2f}M")

    # Optimizer — AdamW with weight decay on non-bias/norm params
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params,   'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    if args.resume and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Save config
    with open(os.path.join(ckpt_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # W&B (optional)
    if args.wandb:
        import wandb
        wandb.init(project='nanogpt-ablation', name=args.run_name, config=vars(args))

    # Training loop
    model.train()
    t0        = time.time()
    log_rows  = []

    x, y = get_batch('train', args.data_dir, args.block_size,
                     args.batch_size, device)

    print(f"\nTraining for {args.max_iters} iterations...")
    print(f"{'iter':>6}  {'train_loss':>11}  {'val_loss':>10}  {'lr':>9}  {'ms/iter':>8}")
    print('-' * 55)

    while iter_num <= args.max_iters:

        # LR schedule
        lr_now = get_lr(iter_num, args.warmup_iters, args.max_iters,
                        args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Eval
        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, args.data_dir, args.block_size,
                                   args.batch_size, args.eval_iters, device)
            t1  = time.time()
            ms  = (t1 - t0) * 1000 / args.eval_interval
            t0  = t1
            print(f"{iter_num:>6}  {losses['train']:>11.4f}  {losses['val']:>10.4f}"
                  f"  {lr_now:>9.2e}  {ms:>7.1f}")

            log_rows.append({'iter': iter_num, **losses, 'lr': lr_now})
            with open(os.path.join(ckpt_dir, 'log.json'), 'w') as f:
                json.dump(log_rows, f, indent=2)

            if args.wandb:
                import wandb
                wandb.log({'train_loss': losses['train'],
                           'val_loss': losses['val'], 'lr': lr_now}, step=iter_num)

            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt = {
                    'model':     model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config':    vars(cfg),
                    'iter_num':  iter_num,
                    'best_val':  best_val,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, 'ckpt.pt'))

        if iter_num == args.max_iters:
            break

        # Forward + backward
        with torch.autocast(device_type=device, dtype=dtype):
            _, loss = model(x, y)

        x, y = get_batch('train', args.data_dir, args.block_size,
                         args.batch_size, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % args.log_interval == 0 and iter_num > 0:
            pass   # detailed logging only at eval_interval

        iter_num += 1

    print(f"\nDone. Best val loss: {best_val:.4f}")
    print(f"Checkpoint: {ckpt_dir}/ckpt.pt")

if __name__ == '__main__':
    main()
