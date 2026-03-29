"""
generate.py — sample text from a trained checkpoint.

Usage:
    python generate.py --checkpoint=checkpoints/baseline/ckpt.pt
    python generate.py --checkpoint=checkpoints/baseline/ckpt.pt --prompt="The neural"
"""

import os
import argparse
import torch
import tiktoken
from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--prompt',     type=str, default='\n')
    p.add_argument('--max_tokens', type=int, default=200)
    p.add_argument('--temperature',type=float, default=0.8)
    p.add_argument('--top_k',      type=int, default=40)
    p.add_argument('--num_samples',type=int, default=3)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt  = torch.load(args.checkpoint, map_location=device)
    cfg   = GPTConfig(**ckpt['config'])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    enc   = tiktoken.get_encoding('gpt2')
    start = enc.encode(args.prompt, allowed_special={'<|endoftext|>'})
    x     = torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0)

    print(f'Model: {cfg.n_layer}L {cfg.n_head}H {cfg.n_embd}d')
    print(f'Prompt: {repr(args.prompt)}')
    print('=' * 60)

    with torch.no_grad():
        for i in range(args.num_samples):
            y = model.generate(x, args.max_tokens,
                               temperature=args.temperature,
                               top_k=args.top_k)
            text = enc.decode(y[0].tolist())
            print(f'\n--- Sample {i+1} ---')
            print(text)

if __name__ == '__main__':
    main()
