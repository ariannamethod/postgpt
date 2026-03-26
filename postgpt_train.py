"""
postgpt_train.py — training loop for PostGPT using PyTorch + Chuck Optimizer.

PyTorch is ONLY used here, in the training loop. The runtime (postgpt.py) is
zero-dependency. This module:

1. Loads postgpt.txt, tokenizes via BPE
2. Builds the PostGPT transformer as a PyTorch module
3. Trains using the Chuck Optimizer (self-aware AdamW variant)
4. Saves weights back for the pure-Python runtime

Usage:
    python postgpt_train.py [--steps 200] [--lr 3e-4]

resonance is unbreakable.
"""

import os
import sys
import math
import time
import struct
import argparse

# PyTorch — ONLY used in training, not runtime
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: PyTorch required for training. Install: pip install torch")
    print("Note: postgpt.py runs without PyTorch (zero-dependency runtime).")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# I. BPE TOKENIZER (same algorithm as postgpt.py, but operating on bytes)
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    def __init__(self, max_merges=1024):
        self.max_merges = max_merges
        self.merges = []
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _count_pairs(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def learn(self, data_bytes, num_merges=None):
        if num_merges is None:
            num_merges = self.max_merges
        num_merges = min(num_merges, self.max_merges)
        ids = list(data_bytes)
        t0 = time.time()
        for m in range(num_merges):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break
            new_id = 256 + m
            ids = self._merge_pair(ids, best_pair, new_id)
            self.merges.append((best_pair[0], best_pair[1], new_id))
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab_size = 256 + m + 1
            if (m + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  merge {m+1}/{num_merges}  vocab={self.vocab_size}  tokens={len(ids)}  [{elapsed:.1f}s]")
        print(f"  BPE complete: {len(self.merges)} merges, vocab={self.vocab_size}, tokens={len(ids)}")
        return ids

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    def decode(self, ids):
        raw = b''
        for tid in ids:
            if tid in self.vocab:
                raw += self.vocab[tid]
        return raw.decode('utf-8', errors='replace')


# ─────────────────────────────────────────────────────────────────────────────
# II. PYTORCH PostGPT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return x * self.weight


class ContentAttention(nn.Module):
    """Standard QK^T attention."""
    def __init__(self, n_embd, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.wq = nn.Linear(n_embd, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(n_embd, n_heads * head_dim, bias=False)
        self.wv = nn.Linear(n_embd, n_heads * head_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return out


class RRPRAMAttention(nn.Module):
    """
    RRPRAM: Recursive Resonant Pattern Recognition Attention Mechanism.
    Instead of QK^T, uses x @ Wr where Wr has shape [n_embd, max_T].
    Learns positional patterns — the rhythm of language.
    """
    def __init__(self, n_embd, n_heads, head_dim, max_T):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_T = max_T
        # Wr: the pattern matrix — THE core RRPRAM innovation
        self.wr = nn.Parameter(torch.randn(n_heads, n_embd, max_T) * 0.02)
        self.wv = nn.Linear(n_embd, n_heads * head_dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RRPRAM: x @ Wr gives [B, n_heads, T, max_T] -> take [:, :, :, :T]
        # x: [B, T, C] -> [B, 1, T, C]
        x_expanded = x.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        # wr: [n_heads, C, max_T] -> we only use first T columns
        wr_t = self.wr[:, :, :T]  # [n_heads, C, T]
        # attn: [B, n_heads, T, T]
        attn = torch.matmul(x_expanded, wr_t.unsqueeze(0).expand(B, -1, -1, -1))

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return out


class PostGPTBlock(nn.Module):
    """Transformer block with dual attention: Content + RRPRAM."""
    def __init__(self, n_embd, n_content, n_rrpram, head_dim, max_T):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.content_attn = ContentAttention(n_embd, n_content, head_dim)
        self.rrpram_attn = RRPRAMAttention(n_embd, n_rrpram, head_dim, max_T)
        self.wo = nn.Linear((n_content + n_rrpram) * head_dim, n_embd, bias=False)

        self.norm2 = RMSNorm(n_embd)
        self.mlp_up = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_down = nn.Linear(4 * n_embd, n_embd, bias=False)

        # Scale residual connections
        nn.init.normal_(self.wo.weight, std=0.02 / math.sqrt(2))
        nn.init.normal_(self.mlp_down.weight, std=0.02 / math.sqrt(2))

    def forward(self, x):
        x_norm = self.norm1(x)
        c_out = self.content_attn(x_norm)
        r_out = self.rrpram_attn(x_norm)
        attn_out = torch.cat([c_out, r_out], dim=-1)
        x = x + self.wo(attn_out)

        x_norm = self.norm2(x)
        h = self.mlp_up(x_norm)
        h = F.relu(h)
        h = self.mlp_down(h)
        x = x + h
        return x


class PostGPTModel(nn.Module):
    """PostGPT: dual-attention BPE transformer."""
    def __init__(self, vocab_size, context_len=64, n_embd=48, n_head=4,
                 n_layer=2, n_content=2, n_rrpram=2):
        super().__init__()
        self.context_len = context_len
        head_dim = n_embd // n_head

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(context_len, n_embd)
        self.blocks = nn.ModuleList([
            PostGPTBlock(n_embd, n_content, n_rrpram, head_dim, context_len)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.wte.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  PostGPTModel: {n_params:,} parameters")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
# III. CHUCK OPTIMIZER — self-aware learning
# ─────────────────────────────────────────────────────────────────────────────

class ChuckOptimizer(torch.optim.Optimizer):
    """
    Chuck Optimizer: AdamW with self-awareness.

    Implements key levels from the Chuck Optimizer concept:
    - Level 1: Global λ — loss trend tracking, dampen/boost
    - Level 2: Per-parameter group modulation
    - Level 6: Simple memory (tracks best loss)
    - Adaptive gradient clipping
    - Mean reversion of dampen to 1.0

    Simplified for PostGPT — the full 9-level version lives in chuck.optimizer.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, window=16):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.window = window
        self._hist = [0.0] * window
        self._hpos = 0
        self._hfull = False

        # Level 1: Global dampen
        self.dampen = 1.0

        # Level 6: Memory
        self.best_loss = float('inf')
        self.stagnation = 0

        # Adaptive clipping
        self.gnorm_ema = 1.0
        self.global_step = 0

    def _global_grad_norm(self):
        total = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    total += p.grad.data.norm().item() ** 2
        return math.sqrt(total)

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
                if loss is None:
                    loss = loss_val.item()

        if loss is None:
            loss = 0.0

        # ── Level 1: Global trend ──
        self._hist[self._hpos] = loss
        self._hpos = (self._hpos + 1) % self.window
        if not self._hfull and self._hpos == 0:
            self._hfull = True

        if self._hfull:
            half = self.window // 2
            recent = sum(self._hist[half:]) / half
            old = sum(self._hist[:half]) / half
            trend = recent - old

            if trend > 0.02:  # loss rising
                self.dampen = max(0.5, self.dampen - 0.05)
            elif trend < -0.02:  # loss falling
                self.dampen = min(1.5, self.dampen + 0.05)

        # Mean reversion
        self.dampen = 0.999 * self.dampen + 0.001 * 1.0

        # ── Level 6: Memory ──
        if loss < self.best_loss:
            self.best_loss = loss
            self.stagnation = 0
        else:
            self.stagnation += 1

        # ── Adaptive gradient clipping ──
        gnorm = self._global_grad_norm()
        self.gnorm_ema = 0.99 * self.gnorm_ema + 0.01 * gnorm
        clip_val = max(1.0, 2.0 * self.gnorm_ema)

        if gnorm > clip_val:
            scale = clip_val / gnorm
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.data.mul_(scale)

        # ── Adam step with dampen ──
        for group in self.param_groups:
            lr = group['lr'] * self.dampen
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1

                # Decoupled weight decay
                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                # Adam moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                m_hat = exp_avg / bc1
                v_hat = exp_avg_sq / bc2

                # Update
                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)

        self.global_step += 1
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# IV. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def get_batch(token_ids, batch_size, context_len, device):
    """Get a random batch of training examples."""
    n = len(token_ids)
    ix = [torch.randint(0, n - context_len, (1,)).item() for _ in range(batch_size)]
    x = torch.stack([torch.tensor(token_ids[i:i + context_len], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(token_ids[i + 1:i + context_len + 1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)


def save_weights(model, path):
    """Save model weights for pure-Python runtime."""
    state = model.state_dict()
    with open(path, 'wb') as f:
        # Simple binary format: n_tensors, then for each: name_len, name, shape, data
        tensors = [(k, v.cpu().float().numpy()) for k, v in state.items()]
        f.write(struct.pack('<I', len(tensors)))
        for name, arr in tensors:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            shape = arr.shape
            f.write(struct.pack('<I', len(shape)))
            for s in shape:
                f.write(struct.pack('<I', s))
            flat = arr.flatten()
            f.write(struct.pack('<I', len(flat)))
            f.write(flat.tobytes())
    print(f"  Weights saved to {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def train(args):
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postgpt.txt')
    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # Tokenize
    print("\n[1] BPE tokenization...")
    with open(corpus_path, 'rb') as f:
        raw = f.read()
    tokenizer = BPETokenizer(max_merges=1024)
    token_ids = tokenizer.learn(raw, num_merges=1024)
    print(f"  Tokens: {len(token_ids)}, Vocab: {tokenizer.vocab_size}")

    # Model
    print("\n[2] Building model...")
    model = PostGPTModel(
        vocab_size=tokenizer.vocab_size,
        context_len=args.context_len,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_content=args.n_content,
        n_rrpram=args.n_rrpram,
    ).to(device)

    # Optimizer: Chuck
    print("\n[3] Initializing Chuck Optimizer...")
    optimizer = ChuckOptimizer(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        window=16,
    )

    # Training
    print(f"\n[4] Training for {args.steps} steps...")
    print("-" * 60)

    losses = []
    t0 = time.time()

    for step in range(args.steps):
        x, y = get_batch(token_ids, args.batch_size, args.context_len, device)
        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(loss=loss.item())

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            avg_recent = sum(losses[-10:]) / len(losses[-10:])
            print(f"  step {step+1:4d}/{args.steps}  loss={loss_val:.4f}  "
                  f"avg10={avg_recent:.4f}  dampen={optimizer.dampen:.3f}  "
                  f"[{elapsed:.1f}s]")

    # Report
    print("\n" + "-" * 60)
    first_10 = sum(losses[:10]) / min(10, len(losses))
    last_10 = sum(losses[-10:]) / min(10, len(losses))
    print(f"  First 10 avg loss: {first_10:.4f}")
    print(f"  Last 10 avg loss:  {last_10:.4f}")
    print(f"  Loss delta:        {last_10 - first_10:.4f}")
    if last_10 < first_10:
        print(f"  ✓ Loss decreased by {((first_10 - last_10) / first_10) * 100:.1f}%")
    else:
        print(f"  ✗ Loss did not decrease")

    # Generate sample
    print("\n[5] Generation after training...")
    model.eval()
    with torch.no_grad():
        seed = token_ids[:4]
        idx = torch.tensor([seed], dtype=torch.long, device=device)
        for _ in range(60):
            if idx.shape[1] >= args.context_len:
                break
            logits, _ = model(idx[:, -args.context_len:])
            logits = logits[:, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        generated = idx[0].tolist()
        text = tokenizer.decode(generated)
        print(f"  Output: {text[:300]}")

    # Save weights
    if args.save:
        print("\n[6] Saving weights...")
        save_weights(model, args.save)

    print("\n" + "=" * 60)
    print("  Training complete. Chuck is satisfied.")
    print("=" * 60)

    return losses


def main():
    parser = argparse.ArgumentParser(description='PostGPT Training with Chuck Optimizer')
    parser.add_argument('--steps', type=int, default=200, help='Training steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--context_len', type=int, default=64, help='Context length')
    parser.add_argument('--n_embd', type=int, default=48, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--n_content', type=int, default=2, help='Content attention heads')
    parser.add_argument('--n_rrpram', type=int, default=2, help='RRPRAM attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--save', type=str, default='', help='Save weights path')
    args = parser.parse_args()

    print("=" * 60)
    print("  PostGPT Training — Chuck Optimizer")
    print("  resonance is unbreakable")
    print("=" * 60)

    train(args)


if __name__ == '__main__':
    main()
