"""
nanoGPT — minimal GPT implementation for ablation study.

Based on: Karpathy (2022), nanoGPT. github.com/karpathy/nanoGPT
Paper:    Vaswani et al. (2017), Attention is All You Need. arXiv:1706.03762

Architecture:
    Token embedding + positional embedding
    -> N x TransformerBlock (CausalSelfAttention + MLP)
    -> LayerNorm -> linear head

Ablation parameters (all in GPTConfig):
    n_layer:  number of transformer blocks       (ablation: depth)
    n_head:   number of attention heads          (ablation: heads)
    n_embd:   embedding dimension                (ablation: width)
    dropout:  dropout rate
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int   = 256    # context length (tokens)
    vocab_size: int   = 50304  # GPT-2 tokenizer: 50257, padded to nearest 64
    n_layer:    int   = 6
    n_head:     int   = 6
    n_embd:     int   = 384
    dropout:    float = 0.1
    bias:       bool  = True   # bias in linear layers and LayerNorms

    @property
    def n_params(self):
        """Approximate parameter count (embedding + transformer blocks)."""
        # token emb + pos emb + blocks + ln_f + head
        return (self.vocab_size * self.n_embd
                + self.block_size * self.n_embd
                + self.n_layer * (
                    # CausalSelfAttention: 4 linears (Q,K,V,proj), each n_embd x n_embd
                    4 * 3 * self.n_embd * self.n_embd
                    # MLP: fc (n_embd -> 4*n_embd) + proj (4*n_embd -> n_embd)
                    + 2 * 4 * self.n_embd * self.n_embd
                    # LayerNorms: 2 x 2 x n_embd (weight + bias)
                    + 4 * self.n_embd
                )
                + self.n_embd           # ln_f
                + self.vocab_size * self.n_embd)  # lm_head


class LayerNorm(nn.Module):
    """LayerNorm with optional bias (torch built-in forces bias=True)."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Key property: each position can only attend to earlier positions.
    Implemented via an upper-triangular mask in the attention matrix.

    For the ablation: varying n_head with fixed n_embd tests whether
    more fine-grained attention (more heads, smaller head_dim) helps.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head

        # Q, K, V projections in one matrix (efficiency)
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: upper triangle = -inf so future tokens are blocked
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
               .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape   # batch, sequence length, embedding dim

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale          # (B, n_head, T, T)
        att   = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att   = F.softmax(att, dim=-1)
        att   = self.attn_drop(att)

        # Weighted sum over values
        y = att @ v                                          # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # reassemble heads

        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """
    Position-wise feed-forward network.
    Hidden dim = 4 * n_embd (standard from Vaswani et al.).
    GELU activation (smoother than ReLU, used in GPT-2).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu  = nn.GELU()
        self.proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.drop  = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.proj(self.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    """
    One transformer block: pre-norm architecture (LayerNorm before attention/MLP).
    Pre-norm is more stable than original post-norm (Xiong et al. 2020).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection around attention
        x = x + self.mlp(self.ln2(x))    # residual connection around MLP
        return x


class GPT(nn.Module):
    """
    GPT language model.

    Forward pass returns (logits, loss) where loss is cross-entropy
    over the next-token prediction task (autoregressive).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte':  nn.Embedding(config.vocab_size, config.n_embd),     # token embeddings
            'wpe':  nn.Embedding(config.block_size, config.n_embd),     # position embeddings
            'drop': nn.Dropout(config.dropout),
            'h':    nn.ModuleList([TransformerBlock(config)
                                   for _ in range(config.n_layer)]),
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        # Language model head: maps embeddings back to vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding and lm_head share weights
        # (Press & Wolf 2017 — reduces parameters, improves performance)
        self.transformer['wte'].weight = self.lm_head.weight

        # Parameter initialisation (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight') or name.endswith('proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx:     (B, T) int64 token indices
        targets: (B, T) int64 next-token targets (optional)
        Returns: (logits (B,T,V), loss or None)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(T, device=idx.device)                    # (T,)
        tok_emb = self.transformer['wte'](idx)                       # (B, T, n_embd)
        pos_emb = self.transformer['wpe'](pos)                       # (T, n_embd)
        x = self.transformer['drop'](tok_emb + pos_emb)

        for block in self.transformer['h']:
            x = block(x)

        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressive generation. idx: (B, T) seed tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_parameters(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
