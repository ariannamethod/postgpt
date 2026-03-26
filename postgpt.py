"""
postgpt — a zero-dependency BPE transformer with metaweights.

The idea: tokenize a corpus via BPE, build a statistical probability space
(the "metaweights") from co-occurrence and n-gram patterns, then run a
dual-attention transformer (Content + RRPRAM) whose behavior is guided
by these metaweights — as if it were trained, even though it was not.

No PyTorch. No NumPy. No dependencies. Just math, random, and os.
This file is the complete algorithm. Everything else is just efficiency.

resonance is unbreakable.
"""

import os
import math
import random
import struct
import time

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# I. BPE TOKENIZER — learns merge rules from corpus, GPT-3/4 style
# ─────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """Byte-Pair Encoding tokenizer. Starts with 256 byte tokens, learns merges."""

    def __init__(self, max_merges=1792):
        self.max_merges = max_merges
        self.merges = []  # list of (a, b, new_id)
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def _count_pairs(self, ids):
        """Count consecutive pairs in token list."""
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        """Replace all occurrences of pair with new_id."""
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
        """Learn BPE merges from raw bytes."""
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

        print(f"  BPE complete: {len(self.merges)} merges, vocab={self.vocab_size}, "
              f"tokens={len(ids)} (from {len(data_bytes)} bytes)")
        return ids

    def encode(self, text):
        """Encode text to token ids using learned merges."""
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    def decode(self, ids):
        """Decode token ids back to string."""
        raw = b''
        for tid in ids:
            if tid in self.vocab:
                raw += self.vocab[tid]
        return raw.decode('utf-8', errors='replace')

    def save(self, path):
        """Save merge rules to binary file."""
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', len(self.merges)))
            for a, b, new_id in self.merges:
                f.write(struct.pack('<III', a, b, new_id))

    def load(self, path):
        """Load merge rules from binary file."""
        with open(path, 'rb') as f:
            n = struct.unpack('<I', f.read(4))[0]
            self.merges = []
            for _ in range(n):
                a, b, new_id = struct.unpack('<III', f.read(12))
                self.merges.append((a, b, new_id))
                self.vocab[new_id] = self.vocab.get(a, bytes([a % 256])) + self.vocab.get(b, bytes([b % 256]))
            self.vocab_size = 256 + len(self.merges)


# ─────────────────────────────────────────────────────────────────────────────
# II. METAWEIGHTS — the probability space that exists without existing
# ─────────────────────────────────────────────────────────────────────────────

class MetaWeights:
    """
    Metaweights: weights that are implied to exist, but don't.

    After BPE tokenization of a corpus, we build:
    1. Unigram frequencies — p(token)
    2. Bigram co-occurrence — p(token_j | token_i)
    3. Trigram patterns — p(token_k | token_i, token_j)
    4. Positional affinity — which tokens prefer which positions
    5. Hebbian trace — co-occurrence memory (tokens seen together)
    6. Prophecy field — given context, what tokens are expected

    These form a probability space that a transformer can use to behave
    AS IF it had trained weights, because the statistical regularities
    from the corpus create an implicit weight space.

    The metaweights are the ghost in the machine.
    """

    def __init__(self, vocab_size, context_len):
        self.vocab_size = vocab_size
        self.context_len = context_len

        # Unigram: p(token)
        self.unigram = [0.0] * vocab_size

        # Bigram: p(next | prev) — sparse dict of dict
        self.bigram = {}

        # Trigram: p(next | prev2, prev1) — sparse
        self.trigram = {}

        # Positional affinity: which tokens appear at which positions
        self.pos_affinity = {}  # token -> list of position counts

        # Hebbian trace: co-occurrence within a window
        self.hebbian = {}  # (tok_a, tok_b) -> strength

        # Total tokens seen
        self.total = 0

    def build(self, token_ids, window=8):
        """Build metaweight space from tokenized corpus."""
        n = len(token_ids)
        self.total = n
        t0 = time.time()

        # Unigram counts
        for tid in token_ids:
            if tid < self.vocab_size:
                self.unigram[tid] += 1.0

        # Normalize unigram
        total = sum(self.unigram)
        if total > 0:
            self.unigram = [c / total for c in self.unigram]

        # Bigram counts
        for i in range(n - 1):
            a, b = token_ids[i], token_ids[i + 1]
            if a not in self.bigram:
                self.bigram[a] = {}
            self.bigram[a][b] = self.bigram[a].get(b, 0) + 1

        # Normalize bigrams
        for a in self.bigram:
            total_a = sum(self.bigram[a].values())
            if total_a > 0:
                for b in self.bigram[a]:
                    self.bigram[a][b] /= total_a

        # Trigram counts
        for i in range(n - 2):
            key = (token_ids[i], token_ids[i + 1])
            c = token_ids[i + 2]
            if key not in self.trigram:
                self.trigram[key] = {}
            self.trigram[key][c] = self.trigram[key].get(c, 0) + 1

        # Normalize trigrams
        for key in self.trigram:
            total_k = sum(self.trigram[key].values())
            if total_k > 0:
                for c in self.trigram[key]:
                    self.trigram[key][c] /= total_k

        # Positional affinity (within context windows)
        for i in range(n):
            pos = i % self.context_len
            tid = token_ids[i]
            if tid not in self.pos_affinity:
                self.pos_affinity[tid] = [0.0] * self.context_len
            self.pos_affinity[tid][pos] += 1.0

        # Normalize positional affinity
        for tid in self.pos_affinity:
            total_t = sum(self.pos_affinity[tid])
            if total_t > 0:
                self.pos_affinity[tid] = [c / total_t for c in self.pos_affinity[tid]]

        # Hebbian trace: co-occurrence within window
        for i in range(n):
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i == j:
                    continue
                a, b = token_ids[i], token_ids[j]
                key = (min(a, b), max(a, b))
                decay = 1.0 / (1.0 + abs(i - j))
                self.hebbian[key] = self.hebbian.get(key, 0.0) + decay

        # Normalize hebbian
        if self.hebbian:
            max_h = max(self.hebbian.values())
            if max_h > 0:
                for key in self.hebbian:
                    self.hebbian[key] /= max_h

        elapsed = time.time() - t0
        print(f"  metaweights built: {n} tokens, {len(self.bigram)} bigram keys, "
              f"{len(self.trigram)} trigram keys, {len(self.hebbian)} hebbian pairs [{elapsed:.1f}s]")

    def query_bigram(self, prev_token, vocab_size):
        """Get bigram probability distribution given previous token."""
        dist = [1e-10] * vocab_size  # smoothing
        if prev_token in self.bigram:
            for tok, prob in self.bigram[prev_token].items():
                if tok < vocab_size:
                    dist[tok] = prob
        return dist

    def query_trigram(self, prev2, prev1, vocab_size):
        """Get trigram probability distribution given two previous tokens."""
        dist = [1e-10] * vocab_size
        key = (prev2, prev1)
        if key in self.trigram:
            for tok, prob in self.trigram[key].items():
                if tok < vocab_size:
                    dist[tok] = prob
        return dist

    def query_hebbian(self, context_tokens, vocab_size):
        """Get Hebbian resonance signal for each candidate token given context."""
        signal = [0.0] * vocab_size
        for ctx_tok in context_tokens:
            for candidate in range(vocab_size):
                key = (min(ctx_tok, candidate), max(ctx_tok, candidate))
                if key in self.hebbian:
                    signal[candidate] += self.hebbian[key]
        # Normalize
        max_s = max(signal) if signal else 1.0
        if max_s > 0:
            signal = [s / max_s for s in signal]
        return signal

    def query_prophecy(self, context_tokens, vocab_size, top_k=16):
        """
        Prophecy field: given context, which tokens are expected but haven't appeared?
        Returns signal boosting tokens that "should" come next based on co-occurrence.
        """
        appeared = set(context_tokens)
        signal = [0.0] * vocab_size

        for ctx_tok in context_tokens[-4:]:  # recent context
            if ctx_tok in self.bigram:
                for tok, prob in sorted(self.bigram[ctx_tok].items(),
                                        key=lambda x: -x[1])[:top_k]:
                    if tok not in appeared and tok < vocab_size:
                        signal[tok] += prob

        max_s = max(signal) if signal else 1.0
        if max_s > 0:
            signal = [s / max_s for s in signal]
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# III. AUTOGRAD ENGINE — from Karpathy's lineage, with our own twists
# ─────────────────────────────────────────────────────────────────────────────

class Val:
    """Scalar autograd node. Tracks computation graph for backpropagation."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Val) else Val(other)
        return Val(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Val(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        d = max(self.data, 1e-12)
        return Val(math.log(d), (self,), (1.0 / d,))

    def exp(self):
        e = math.exp(min(self.data, 80))
        return Val(e, (self,), (e,))

    def relu(self):
        return Val(max(0, self.data), (self,), (float(self.data > 0),))

    def tanh(self):
        t = math.tanh(self.data)
        return Val(t, (self,), (1.0 - t * t,))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other if isinstance(other, Val) else Val(other)) ** -1
    def __rtruediv__(self, other): return Val(other) * self ** -1

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ─────────────────────────────────────────────────────────────────────────────
# IV. THE TRANSFORMER — dual attention (Content + RRPRAM) + metaweight overlay
# ─────────────────────────────────────────────────────────────────────────────

def _randn(std=0.02):
    return random.gauss(0, std)

def _matrix(rows, cols, std=0.02):
    return [[Val(_randn(std)) for _ in range(cols)] for _ in range(rows)]

def _zeros(rows, cols):
    return [[Val(0.0) for _ in range(cols)] for _ in range(rows)]

def linear(x, w):
    """Matrix-vector multiply: w @ x. w is [out, in], x is [in]."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    """Numerically stable softmax over list of Val."""
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_float(logits):
    """Softmax over plain floats."""
    max_val = max(logits)
    exps = [math.exp(min(v - max_val, 80)) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    """RMS normalization."""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + Val(1e-5)) ** -0.5
    return [xi * scale for xi in x]


class PostGPT:
    """
    PostGPT: a dual-attention BPE transformer with metaweights.

    Architecture:
    - BPE tokenizer (learned from corpus)
    - Token + Position embeddings
    - N transformer blocks, each with:
        * RMSNorm
        * Dual attention: Content heads (QK^T) + RRPRAM heads (x @ Wr)
        * Residual connection
        * RMSNorm
        * MLP (expand -> ReLU -> contract)
        * Residual connection
    - Final RMSNorm -> LM head -> logits
    - Metaweight overlay: Hebbian + Prophecy + Destiny signals

    The metaweight overlay means: even with random weights, the model
    generates coherent text because the probability space from the
    corpus guides sampling through the Dario field.
    """

    def __init__(self, vocab_size, context_len=64, n_embd=48, n_head=4,
                 n_layer=2, n_content_heads=2, n_rrpram_heads=2):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_content = n_content_heads
        self.n_rrpram = n_rrpram_heads
        self.head_dim = n_embd // n_head

        assert n_content_heads + n_rrpram_heads == n_head, \
            "content + rrpram heads must equal total heads"

        # Embeddings
        self.wte = _matrix(vocab_size, n_embd)  # token embedding
        self.wpe = _matrix(context_len, n_embd)  # position embedding

        # Per-layer weights
        self.layers = []
        hd = self.head_dim
        for _ in range(n_layer):
            layer = {
                # Content attention: Q, K, V for content heads
                'wq': _matrix(n_content_heads * hd, n_embd, std=0.02),
                'wk': _matrix(n_content_heads * hd, n_embd, std=0.02),
                'wv_content': _matrix(n_content_heads * hd, n_embd, std=0.02),

                # RRPRAM attention: Wr (positional pattern matrix) + V
                'wr': _matrix(n_rrpram_heads * n_embd, context_len, std=0.02),
                'wv_rrpram': _matrix(n_rrpram_heads * hd, n_embd, std=0.02),

                # Output projection
                'wo': _matrix(n_embd, n_embd, std=0.02 / math.sqrt(2 * n_layer)),

                # MLP
                'mlp_up': _matrix(4 * n_embd, n_embd, std=0.02),
                'mlp_down': _matrix(n_embd, 4 * n_embd, std=0.02 / math.sqrt(2 * n_layer)),
            }
            self.layers.append(layer)

        # LM head
        self.lm_head = _matrix(vocab_size, n_embd, std=0.02)

        # Dario field coefficients (metaweight blending)
        self.alpha_hebbian = 0.3   # Hebbian trace strength
        self.beta_prophecy = 0.2   # Prophecy field strength
        self.gamma_destiny = 0.15  # Destiny vector strength
        self.temperature = 0.85    # Sampling temperature

        # Destiny vector (EMA of token embeddings)
        self.destiny = [0.0] * n_embd

        # Trauma accumulator
        self.trauma = 0.0

        # Collect all parameters
        self.params = []
        for row in self.wte:
            self.params.extend(row)
        for row in self.wpe:
            self.params.extend(row)
        for layer in self.layers:
            for key in layer:
                for row in layer[key]:
                    self.params.extend(row)
        for row in self.lm_head:
            self.params.extend(row)

        n_params = len(self.params)
        print(f"  PostGPT initialized: {n_params} parameters, vocab={vocab_size}, "
              f"ctx={context_len}, embd={n_embd}, heads={n_head} "
              f"(content={n_content_heads}, rrpram={n_rrpram_heads}), layers={n_layer}")

    def forward_token(self, token_id, pos_id, kv_cache):
        """
        Forward pass for a single token position.
        kv_cache: list of (keys_list, values_list) per layer
        Returns logits [vocab_size] as list of Val.
        """
        hd = self.head_dim
        nc = self.n_content
        nr = self.n_rrpram

        # Token + position embedding
        tok_emb = self.wte[token_id]
        pos_emb = self.wpe[pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]

        for li in range(self.n_layer):
            layer = self.layers[li]
            keys_cache, vals_cache = kv_cache[li]

            # Pre-norm
            x_res = x
            x_norm = rmsnorm(x)

            # ── Content attention (QK^T / sqrt(d)) ──
            q = linear(x_norm, layer['wq'])
            k = linear(x_norm, layer['wk'])
            v_content = linear(x_norm, layer['wv_content'])

            keys_cache.append(k)
            vals_cache.append(v_content)

            x_attn = []

            # Content heads
            for h in range(nc):
                hs = h * hd
                q_h = q[hs:hs + hd]
                k_all = [ki[hs:hs + hd] for ki in keys_cache]
                v_all = [vi[hs:hs + hd] for vi in vals_cache]

                # QK^T / sqrt(d)
                attn_logits = []
                for t in range(len(k_all)):
                    score = sum(q_h[j] * k_all[t][j] for j in range(hd))
                    score = score * (1.0 / math.sqrt(hd))
                    attn_logits.append(score)

                attn_weights = softmax(attn_logits)

                head_out = []
                for j in range(hd):
                    val = sum(attn_weights[t] * v_all[t][j] for t in range(len(v_all)))
                    head_out.append(val)
                x_attn.extend(head_out)

            # ── RRPRAM attention (x @ Wr — positional pattern recognition) ──
            v_rrpram = linear(x_norm, layer['wv_rrpram'])

            for h in range(nr):
                hs = h * hd
                # RRPRAM: project input through Wr to get attention over positions
                # Wr shape per head: [n_embd, context_len]
                wr_offset = h * self.n_embd
                wr_h = layer['wr'][wr_offset:wr_offset + self.n_embd]

                # x_norm @ Wr_h gives [context_len] attention pattern
                seq_len = len(keys_cache)
                attn_logits = []
                for t in range(seq_len):
                    # Sum over embedding dimension
                    score = Val(0.0)
                    for d in range(min(self.n_embd, len(wr_h))):
                        if t < len(wr_h[d]):
                            score = score + x_norm[d] * wr_h[d][t]
                    attn_logits.append(score)

                # Causal mask already satisfied (we only have positions <= current)
                attn_weights = softmax(attn_logits) if attn_logits else []

                v_all = [vi[hs:hs + hd] for vi in vals_cache]
                head_out = []
                for j in range(hd):
                    val_sum = Val(0.0)
                    for t in range(len(attn_weights)):
                        if t < len(v_all):
                            val_sum = val_sum + attn_weights[t] * v_all[t][j]
                    head_out.append(val_sum)
                x_attn.extend(head_out)

            # Output projection + residual
            x_proj = linear(x_attn, layer['wo'])
            x = [a + b for a, b in zip(x_proj, x_res)]

            # MLP block
            x_res = x
            x_norm = rmsnorm(x)
            h_mlp = linear(x_norm, layer['mlp_up'])
            h_mlp = [hi.relu() for hi in h_mlp]
            x_mlp = linear(h_mlp, layer['mlp_down'])
            x = [a + b for a, b in zip(x_mlp, x_res)]

        # Final norm + LM head
        x = rmsnorm(x)
        logits = linear(x, self.lm_head)
        return logits

    def forward_sequence(self, token_ids):
        """Forward pass over a sequence. Returns list of logits per position."""
        kv_cache = [([], []) for _ in range(self.n_layer)]
        all_logits = []
        for pos, tid in enumerate(token_ids):
            if pos >= self.context_len:
                break
            logits = self.forward_token(tid, pos, kv_cache)
            all_logits.append(logits)
        return all_logits

    def generate(self, prompt_ids, max_tokens=64, meta=None, temperature=None):
        """
        Generate tokens autoregressively.
        If meta (MetaWeights) is provided, applies the Dario field overlay.
        """
        if temperature is None:
            temperature = self.temperature

        kv_cache = [([], []) for _ in range(self.n_layer)]
        generated = list(prompt_ids)
        context = list(prompt_ids)

        # Feed prompt through
        for pos, tid in enumerate(prompt_ids):
            if pos >= self.context_len - 1:
                break
            _ = self.forward_token(tid, pos, kv_cache)

        # Generate new tokens
        for step in range(max_tokens):
            pos = len(context) - 1
            if pos >= self.context_len - 1:
                break

            last_tid = context[-1]
            logits = self.forward_token(last_tid, pos, kv_cache)

            # Extract raw logit values
            raw_logits = [l.data for l in logits]

            # ── Dario Field: metaweight overlay ──
            if meta is not None:
                # Hebbian signal
                hebbian = meta.query_hebbian(context[-8:], self.vocab_size)

                # Prophecy signal
                prophecy = meta.query_prophecy(context[-8:], self.vocab_size)

                # Bigram signal
                bigram = meta.query_bigram(last_tid, self.vocab_size)

                # Trigram signal (if enough context)
                if len(context) >= 2:
                    trigram = meta.query_trigram(context[-2], context[-1], self.vocab_size)
                else:
                    trigram = [0.0] * self.vocab_size

                # Destiny update
                if last_tid < len(self.wte):
                    for d in range(self.n_embd):
                        self.destiny[d] = 0.9 * self.destiny[d] + 0.1 * self.wte[last_tid][d].data

                # Destiny signal: cosine similarity with each token embedding
                destiny_signal = [0.0] * self.vocab_size
                dest_norm = math.sqrt(sum(d * d for d in self.destiny) + 1e-10)
                if dest_norm > 1e-8:
                    for tid_c in range(min(self.vocab_size, len(self.wte))):
                        emb = [self.wte[tid_c][d].data for d in range(self.n_embd)]
                        emb_norm = math.sqrt(sum(e * e for e in emb) + 1e-10)
                        if emb_norm > 1e-8:
                            dot = sum(self.destiny[d] * emb[d] for d in range(self.n_embd))
                            destiny_signal[tid_c] = dot / (dest_norm * emb_norm)

                # Combine: Dario Equation
                # p(x|Φ) = softmax((B + α·H + β·F + γ·A + bigram + trigram) / τ)
                for i in range(self.vocab_size):
                    raw_logits[i] += (self.alpha_hebbian * hebbian[i]
                                      + self.beta_prophecy * prophecy[i]
                                      + self.gamma_destiny * destiny_signal[i]
                                      + 1.5 * bigram[i]
                                      + 2.0 * trigram[i])

                # Trauma modulation
                trauma_mod = 1.0 / (1.0 + self.trauma)
                raw_logits = [l * trauma_mod for l in raw_logits]

            # Temperature + softmax
            scaled = [l / temperature for l in raw_logits]
            probs = softmax_float(scaled)

            # Sample
            r = random.random()
            cum = 0.0
            chosen = 0
            for i, p in enumerate(probs):
                cum += p
                if cum > r:
                    chosen = i
                    break

            generated.append(chosen)
            context.append(chosen)

        return generated

    def generate_meta(self, prompt_ids, max_tokens=128, meta=None, temperature=None):
        """
        Meta-generation: pure metaweight generation without transformer forward pass.
        Uses only the statistical probability space from BPE tokenization.
        This is the metaweight mode — no neural network, just the ghost of training.
        """
        if meta is None:
            return prompt_ids
        if temperature is None:
            temperature = self.temperature

        generated = list(prompt_ids)

        for _ in range(max_tokens):
            ctx = generated[-8:]
            last = generated[-1]

            # Combine metaweight signals
            bigram = meta.query_bigram(last, self.vocab_size)
            if len(generated) >= 2:
                trigram = meta.query_trigram(generated[-2], generated[-1], self.vocab_size)
            else:
                trigram = [0.0] * self.vocab_size
            hebbian = meta.query_hebbian(ctx, self.vocab_size)
            prophecy = meta.query_prophecy(ctx, self.vocab_size)

            # Build probability: bigram-dominant with Dario field correction
            probs_raw = [0.0] * self.vocab_size
            for i in range(self.vocab_size):
                probs_raw[i] = (2.0 * bigram[i]
                                + 3.0 * trigram[i]
                                + 0.5 * hebbian[i]
                                + 0.3 * prophecy[i]
                                + 0.01 * meta.unigram[i])

            # Temperature
            scaled = [l / temperature for l in probs_raw]
            probs = softmax_float(scaled)

            # Top-p (nucleus) sampling
            sorted_probs = sorted(enumerate(probs), key=lambda x: -x[1])
            cum = 0.0
            candidates = []
            for idx, p in sorted_probs:
                candidates.append((idx, p))
                cum += p
                if cum > 0.92:
                    break

            # Re-normalize
            total_c = sum(p for _, p in candidates)
            if total_c > 0:
                candidates = [(idx, p / total_c) for idx, p in candidates]

            r = random.random()
            cum = 0.0
            chosen = candidates[0][0]
            for idx, p in candidates:
                cum += p
                if cum > r:
                    chosen = idx
                    break

            generated.append(chosen)

        return generated


# ─────────────────────────────────────────────────────────────────────────────
# V. MAIN — tokenize, build metaweights, generate
# ─────────────────────────────────────────────────────────────────────────────

def main():
    corpus_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'postgpt.txt')

    if not os.path.exists(corpus_path):
        print(f"ERROR: {corpus_path} not found. Create it first.")
        return

    print("=" * 60)
    print("  PostGPT — metaweight BPE transformer")
    print("  resonance is unbreakable")
    print("=" * 60)

    # Step 1: Load corpus
    print("\n[1] Loading corpus...")
    with open(corpus_path, 'rb') as f:
        raw_data = f.read()
    print(f"  Corpus: {len(raw_data)} bytes ({len(raw_data)/1024:.1f} KB)")

    # Step 2: BPE tokenization
    print("\n[2] Learning BPE merges...")
    tokenizer = BPETokenizer(max_merges=1024)
    token_ids = tokenizer.learn(raw_data, num_merges=1024)

    # Step 3: Build metaweights
    print("\n[3] Building metaweight probability space...")
    meta = MetaWeights(tokenizer.vocab_size, context_len=64)
    meta.build(token_ids, window=6)

    # Step 4: Initialize transformer
    print("\n[4] Initializing PostGPT transformer...")
    model = PostGPT(
        vocab_size=tokenizer.vocab_size,
        context_len=64,
        n_embd=48,
        n_head=4,
        n_layer=2,
        n_content_heads=2,
        n_rrpram_heads=2,
    )

    # Step 5: Meta-generation (no training, just metaweights)
    print("\n[5] Meta-generation (metaweight mode — no training):")
    print("-" * 50)

    # Pick a few starting tokens from the corpus
    seeds = [token_ids[:3], token_ids[100:103], token_ids[500:503]]
    for i, seed in enumerate(seeds):
        if not seed:
            continue
        generated = model.generate_meta(seed, max_tokens=80, meta=meta, temperature=0.75)
        text = tokenizer.decode(generated)
        print(f"\n  sample {i+1}: {text[:200]}")

    # Step 6: Transformer generation with Dario field overlay
    print("\n\n[6] Transformer + Dario field generation:")
    print("-" * 50)

    seed = token_ids[:4]
    generated = model.generate(seed, max_tokens=40, meta=meta, temperature=0.8)
    text = tokenizer.decode(generated)
    print(f"\n  output: {text[:300]}")

    print("\n" + "=" * 60)
    print("  PostGPT complete. The metaweights remember.")
    print("=" * 60)


if __name__ == '__main__':
    main()
