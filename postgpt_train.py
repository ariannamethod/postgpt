"""
postgpt_train.py — training loop for PostGPT on notorch + the Chuck Optimizer.

NO PyTorch. The training path now runs on notorch (pure-C tensor/autograd library)
through the `ariannamethod/` ctypes shim. The runtime (postgpt.py) was always
zero-dependency; the training loop finally matches its own philosophy — the only
dependency it ever carried, the 2.7 GB of torch, is gone.

The PostGPT graph (dual attention: Content QK^T + RRPRAM x@Wr, RMSNorm, ReLU MLP,
tied lm_head) is built on the notorch autograd tape — every op is the same math
the PyTorch version expressed, verified bit-parity against the notorch C kernels.
Chuck is the real `nt_tape_chuck_step`, not a torch.optim re-implementation.

This module:
1. Loads postgpt.txt, tokenizes via BPE (pure Python)
2. Builds the PostGPT parameters as notorch tensors
3. Trains through the notorch tape with the Chuck Optimizer
4. Saves weights via notorch's binary format

Usage:
    python postgpt_train.py [--steps 200] [--lr 3e-4]

resonance is unbreakable.
"""

import os
import sys
import math
import time
import random
import argparse

# notorch — replaces PyTorch for the whole training path. No torch import anywhere.
try:
    from ariannamethod import (
        Tensor, NotorchEngine, ChuckOptimizer,
        softmax, multinomial, seed,
    )
except Exception as e:
    print(f"ERROR: notorch shim failed to load: {e}")
    print("The shim compiles ariannamethod/notorch.c on first run (needs a C compiler).")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# I. BPE TOKENIZER (same algorithm as postgpt.py, operating on bytes)
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
# II. PostGPT PARAMETERS on notorch
# ─────────────────────────────────────────────────────────────────────────────
#
# Flat, ordered parameter list consumed by NotorchEngine (lm_head is TIED to wte):
#   [wte, wpe,
#    (rms1, c_wq, c_wk, c_wv, r_wr, r_wv, wo, rms2, mlp_up, mlp_down) * n_layers,
#    norm_f]

def build_params(vocab, cfg):
    dim = cfg['dim']; hd = cfg['head_dim']
    nc = cfg['n_content']; nr = cfg['n_rrpram']
    L = cfg['n_layers']; ctx = cfg['ctx']
    P = []

    def linear(in_f, out_f):
        t = Tensor.zeros((out_f, in_f)); t.xavier_(in_f, out_f); return t  # bias-free

    # embeddings
    wte = Tensor.zeros((vocab, dim)); wte.rand_(0.02); P.append(wte)
    wpe = Tensor.zeros((ctx, dim));   wpe.rand_(0.02); P.append(wpe)

    for _ in range(L):
        P.append(Tensor.ones(dim))                       # rms1
        P.append(linear(dim, nc * hd))                   # content wq
        P.append(linear(dim, nc * hd))                   # content wk
        P.append(linear(dim, nc * hd))                   # content wv
        wr = Tensor.zeros(nr * dim * ctx); wr.rand_(0.02)  # RRPRAM Wr [nr, dim, ctx] flat
        P.append(wr)
        P.append(linear(dim, nr * hd))                   # rrpram wv
        P.append(linear((nc + nr) * hd, dim))            # wo
        P.append(Tensor.ones(dim))                       # rms2
        P.append(linear(dim, 4 * dim))                   # mlp_up
        P.append(linear(4 * dim, dim))                   # mlp_down

    P.append(Tensor.ones(dim))                           # norm_f
    return P


# ─────────────────────────────────────────────────────────────────────────────
# III. DATA
# ─────────────────────────────────────────────────────────────────────────────

def get_sequence(token_ids, context_len):
    """One random training sequence: (input[ctx], target[ctx]) shifted by one."""
    n = len(token_ids)
    i = random.randint(0, n - context_len - 1)
    x = token_ids[i:i + context_len]
    y = token_ids[i + 1:i + context_len + 1]
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# IV. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postgpt.txt')
    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found")
        return

    seed(42)
    random.seed(42)

    # Tokenize
    print("\n[1] BPE tokenization...")
    with open(corpus_path, 'rb') as f:
        raw = f.read()
    tokenizer = BPETokenizer(max_merges=1024)
    token_ids = tokenizer.learn(raw, num_merges=1024)
    print(f"  Tokens: {len(token_ids)}, Vocab: {tokenizer.vocab_size}")

    cfg = {
        'ctx': args.context_len,
        'dim': args.n_embd,
        'head_dim': args.n_embd // args.n_head,
        'n_content': args.n_content,
        'n_rrpram': args.n_rrpram,
        'n_layers': args.n_layer,
        'vocab': tokenizer.vocab_size,
    }

    # Model
    print("\n[2] Building model (notorch)...")
    params = build_params(tokenizer.vocab_size, cfg)
    n_params = sum(p.numel for p in params)
    print(f"  PostGPT (notorch): {n_params:,} parameters, {len(params)} tensors")

    # Optimizer: Chuck (the real nt_tape_chuck_step)
    print("\n[3] Initializing Chuck Optimizer (notorch)...")
    optimizer = ChuckOptimizer(lr=args.lr, max_grad_norm=1.0)
    engine = NotorchEngine(params, cfg, optimizer)

    # Training (one sequence per step — the notorch tape is per-sequence)
    print(f"\n[4] Training for {args.steps} steps...")
    print("-" * 60)
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        x, y = get_sequence(token_ids, args.context_len)
        loss_val = engine.step(x, y)
        losses.append(loss_val)
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            avg_recent = sum(losses[-10:]) / len(losses[-10:])
            print(f"  step {step+1:4d}/{args.steps}  loss={loss_val:.4f}  "
                  f"avg10={avg_recent:.4f}  [{elapsed:.1f}s]")

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
    window = token_ids[:4]
    generated = list(window)
    for _ in range(60):
        if len(generated) >= args.context_len:
            break
        logits = engine.logits_last(generated[-args.context_len:])
        probs = softmax(logits, temperature=0.8)
        generated.append(multinomial(probs))
    text = tokenizer.decode(generated)
    print(f"  Output: {text[:300]}")

    # Save weights (notorch binary)
    if args.save:
        print("\n[6] Saving weights...")
        engine.save(args.save)
        print(f"  Weights saved to {args.save} ({os.path.getsize(args.save) / 1024:.1f} KB)")

    print("\n" + "=" * 60)
    print("  Training complete. Chuck is satisfied. No torch was harmed.")
    print("=" * 60)
    return losses


def main():
    parser = argparse.ArgumentParser(description='PostGPT Training on notorch + Chuck Optimizer')
    parser.add_argument('--steps', type=int, default=200, help='Training steps')
    parser.add_argument('--context_len', type=int, default=64, help='Context length')
    parser.add_argument('--n_embd', type=int, default=48, help='Embedding dimension')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layer', type=int, default=2, help='Number of layers')
    parser.add_argument('--n_content', type=int, default=2, help='Content attention heads')
    parser.add_argument('--n_rrpram', type=int, default=2, help='RRPRAM attention heads')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--save', type=str, default='', help='Save weights path')
    args = parser.parse_args()

    print("=" * 60)
    print("  PostGPT Training — notorch + Chuck Optimizer")
    print("  resonance is unbreakable")
    print("=" * 60)

    train(args)


if __name__ == '__main__':
    main()
