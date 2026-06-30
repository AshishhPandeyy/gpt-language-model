"""
train.py — GPT training on openwebtext_10gb.txt (or any large text file)

Uses a streaming on-disk data loader that reads random byte-offset chunks
directly from the file, so the 10 GB dataset never fully lives in RAM.

Run:
    python train.py
"""

import os
import math
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_FILE   = "openwebtext_10gb.txt"
VAL_SPLIT    = 0.05          # hold out last 5% of file bytes for validation
MODEL_OUT    = "model-owt.pkl"
CHECKPOINT   = "checkpoint.pt"  # intermediate checkpoint (state-dict based)

# Model hyper-params
batch_size   = 32
block_size   = 128           # context length (bump to 256/512 when VRAM allows)
n_embd       = 384
n_head       = 6
n_layer      = 6
dropout      = 0.1

# Training hyper-params
max_iters     = 5000
eval_iters    = 100           # batches averaged per eval
eval_interval = 500           # evaluate every N steps
learning_rate = 3e-4
grad_clip     = 1.0
save_interval = 1000          # save checkpoint every N steps

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Using device: {device}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode_ordinary(s)   # encode_ordinary skips special tokens

# ---------------------------------------------------------------------------
# Streaming data loader
# ---------------------------------------------------------------------------
# Strategy: the file is one giant blob of UTF-8 text.
# We split it into a "train region" and "val region" by byte offset.
# get_batch() picks a random byte offset within the region, reads a chunk
# of bytes large enough to tokenise at least (block_size + 1) tokens, then
# randomly picks a starting position within that chunk.

CHUNK_BYTES = block_size * 6   # ~6 bytes/token average; we grab more than needed

file_size = os.path.getsize(TRAIN_FILE)
val_start = int(file_size * (1 - VAL_SPLIT))   # byte offset where val region begins

print(f"File size  : {file_size / 1e9:.2f} GB")
print(f"Train bytes: {val_start / 1e9:.2f} GB  |  Val bytes: {(file_size - val_start) / 1e6:.1f} MB")


def read_chunk(file_obj, byte_start: int, region_end: int) -> list:
    """
    Seek to byte_start, read a window, decode it, tokenise, return token list.
    We over-read so we always get enough tokens even after UTF-8 boundary trimming.
    """
    read_size = min(CHUNK_BYTES * 4, region_end - byte_start)
    file_obj.seek(byte_start)
    raw = file_obj.read(read_size)

    # Trim to valid UTF-8 (we may have landed mid-codepoint)
    raw = raw.decode("utf-8", errors="ignore")

    tokens = encode(raw)
    return tokens


_file_handle = open(TRAIN_FILE, "rb")   # single shared file handle


def get_batch(split: str):
    """Return a (x, y) batch sampled from disk without loading the full file."""
    if split == "train":
        region_start, region_end = 0, val_start
    else:
        region_start, region_end = val_start, file_size

    x_list, y_list = [], []

    while len(x_list) < batch_size:
        # Pick a random byte offset inside the region
        max_start = region_end - CHUNK_BYTES * 4
        if max_start <= region_start:
            byte_start = region_start
        else:
            byte_start = random.randint(region_start, max_start)

        tokens = read_chunk(_file_handle, byte_start, region_end)

        if len(tokens) < block_size + 1:
            continue   # chunk too short at end of file — retry

        # Pick a random token offset inside the decoded chunk
        max_tok_start = len(tokens) - block_size - 1
        tok_start = random.randint(0, max_tok_start)

        x_list.append(tokens[tok_start : tok_start + block_size])
        y_list.append(tokens[tok_start + 1 : tok_start + block_size + 1])

    x = torch.tensor(x_list, dtype=torch.long, device=device)
    y = torch.tensor(y_list, dtype=torch.long, device=device)
    return x, y


# ---------------------------------------------------------------------------
# Model definition (identical to v1.ipynb)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with fused QKV projection"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.drop   = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.drop(out)
        return out


class FeedForward(nn.Module):
    """FFN with GELU activation"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block with Pre-RMSNorm"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2  = RMSNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f    = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        self.token_embedding_table.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # GPT-2 style residual projection init
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, index, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


# ---------------------------------------------------------------------------
# Resume from checkpoint if available
# ---------------------------------------------------------------------------
start_iter    = 0
best_val_loss = float('inf')

model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate,
    betas=(0.9, 0.95), weight_decay=0.1
)

if os.path.exists(CHECKPOINT):
    print(f"Resuming from {CHECKPOINT}...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_iter    = ckpt['iter'] + 1
    best_val_loss = ckpt.get('best_val_loss', float('inf'))
    print(f"  Resumed at iter {start_iter}, best val loss: {best_val_loss:.4f}")
else:
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---------------------------------------------------------------------------
# LR schedule (linear warmup + cosine decay)
# ---------------------------------------------------------------------------
def get_lr(it):
    warmup_iters = 200
    min_lr = learning_rate / 10
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print(f"\nStarting training from iter {start_iter} to {max_iters} ...")
print(f"{'iter':>6}  {'train':>8}  {'val':>8}  {'lr':>10}")
print("-" * 40)

for it in range(start_iter, max_iters):
    # Update LR
    lr = get_lr(it)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # Evaluate periodically
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"{it:6d}  {losses['train']:8.4f}  {losses['val']:8.4f}  {lr:10.2e}")
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']

    # Save checkpoint periodically
    if it > 0 and it % save_interval == 0:
        ckpt = {
            'iter'          : it,
            'model'         : model.state_dict(),
            'optimizer'     : optimizer.state_dict(),
            'best_val_loss' : best_val_loss,
        }
        torch.save(ckpt, CHECKPOINT)
        print(f"  Checkpoint saved at iter {it}")

    # Forward + backward pass
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

# ---------------------------------------------------------------------------
# Final evaluation and save
# ---------------------------------------------------------------------------
losses = estimate_loss()
print(f"\nFinal — train: {losses['train']:.4f} | val: {losses['val']:.4f}")

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {MODEL_OUT}")

_file_handle.close()
