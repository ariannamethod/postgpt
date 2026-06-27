"""
notorch_nn.py — PostGPT neural network API backed by libnotorch (ctypes).

Drop-in replacement for the PyTorch training path. No PyTorch. No numpy.
Just ctypes to the C library that PostGPT's metaweights gave birth to (SPA, RRPRAM,
the Dario field all flowed from here into notorch). The runtime (postgpt.py) was
always zero-dependency; now training is too.

The PostGPT graph (dual attention: Content QK^T + RRPRAM x@Wr) is built on the
notorch autograd tape — every op verified bit-parity to postgpt_train.py:
  Content attention -> nt_mh_causal_attention (scale 1/sqrt(head_dim))
  RRPRAM            -> nt_rrpram_attention    (scores = x_i . Wr[h][:,j], no scale)
  concat            -> nt_concat
  RMSNorm/Linear    -> nt_seq_rmsnorm / nt_seq_linear
  MLP ReLU          -> nt_relu
  output (tied)     -> nt_seq_linear on the wte tensor
  loss              -> nt_seq_cross_entropy
Optimizer: Chuck (nt_tape_chuck_step) — the real one, not a torch.optim re-impl.

Usage:
    from ariannamethod.notorch_nn import *
"""

import ctypes
import os
import math
import struct
import random
import platform

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD libnotorch (compile from the vendored notorch.c on first run)
# ═══════════════════════════════════════════════════════════════════════════════

_dir = os.path.dirname(os.path.abspath(__file__))

def _find_or_build_lib():
    for ext in ('.dylib', '.so', '.dll'):
        p = os.path.join(_dir, f'libnotorch{ext}')
        if os.path.exists(p):
            return p
    src = os.path.join(_dir, 'notorch.c')
    if not os.path.exists(src):
        raise RuntimeError(f'notorch.c not found in {_dir}')
    is_darwin = platform.system() == 'Darwin'
    out = os.path.join(_dir, 'libnotorch.dylib' if is_darwin else 'libnotorch.so')
    import subprocess
    base = ['cc', '-O2', '-std=c11', '-shared', '-fPIC', '-o', out, src]
    # Try the accelerated path first (Accelerate on Apple Silicon / OpenBLAS on
    # Linux), fall back to portable scalar if the BLAS link fails.
    attempts = []
    if is_darwin:
        attempts.append(base + ['-DUSE_BLAS', '-DACCELERATE', '-DACCELERATE_NEW_LAPACK',
                                '-framework', 'Accelerate', '-lm'])
    else:
        attempts.append(base + ['-DUSE_BLAS', '-lopenblas', '-lm'])
    attempts.append(base + ['-lm'])  # portable scalar fallback
    last_err = None
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return out
        except subprocess.CalledProcessError as e:
            last_err = e.stderr.decode('utf-8', 'replace') if e.stderr else str(e)
    raise RuntimeError(f'failed to build libnotorch:\n{last_err}')

_lib = ctypes.CDLL(_find_or_build_lib())

# ═══════════════════════════════════════════════════════════════════════════════
# C FUNCTION SIGNATURES
# ═══════════════════════════════════════════════════════════════════════════════

_V = ctypes.c_void_p
_I = ctypes.c_int
_F = ctypes.c_float

def _sig(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)

# tensor
_sig('nt_tensor_new', _V, _I)
_sig('nt_tensor_new2d', _V, _I, _I)
_sig('nt_tensor_free', None, _V)
_sig('nt_tensor_fill', None, _V, _F)
_sig('nt_tensor_xavier', None, _V, _I, _I)
_sig('nt_tensor_rand', None, _V, _F)
_sig('nt_seed', None, ctypes.c_uint64)
# tape
_sig('nt_tape_start', None)
_sig('nt_tape_clear', None)
_sig('nt_tape_get', _V)
_sig('nt_tape_param', _I, _V)
_sig('nt_tape_no_decay', None, _I)
_sig('nt_tape_backward', None, _I)
_sig('nt_tape_clip_grads', _F, _F)
_sig('nt_tape_chuck_step', None, _F, _F)
_sig('nt_tape_record', _I, _V, _I, _I, _I, _F)
_sig('nt_train_mode', None, _I)
# forward ops
_sig('nt_seq_embedding', _I, _I, _I, _I, _I, _I)
_sig('nt_seq_rmsnorm', _I, _I, _I, _I, _I)
_sig('nt_seq_linear', _I, _I, _I, _I)
_sig('nt_mh_causal_attention', _I, _I, _I, _I, _I, _I)
_sig('nt_rrpram_attention', _I, _I, _I, _I, _I, _I, _I, _I)
_sig('nt_concat', _I, _I, _I, _I)
_sig('nt_relu', _I, _I)
_sig('nt_add', _I, _I, _I)
_sig('nt_seq_cross_entropy', _I, _I, _I, _I, _I)
# save/load
_sig('nt_save', _I, ctypes.c_char_p, ctypes.POINTER(_V), _I)
_sig('nt_load', ctypes.POINTER(_V), ctypes.c_char_p, ctypes.POINTER(_I))

# ═══════════════════════════════════════════════════════════════════════════════
# C STRUCTS — exact layout (no USE_CUDA fields: the shim lib is CPU-only)
# ═══════════════════════════════════════════════════════════════════════════════

class _NtTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(_F)),
        ("ndim", _I),
        ("shape", _I * 8),
        ("stride", _I * 8),
        ("len", _I),
        ("refcount", _I),
    ]

# Must match notorch.h nt_tape_entry EXACTLY (incl. trailing `frozen`), or the
# entries[idx] stride is wrong and loss/logits reads land in the wrong slot.
class _NtTapeEntry(ctypes.Structure):
    _fields_ = [
        ("output", _V), ("grad", _V),
        ("op", _I), ("parent1", _I), ("parent2", _I), ("parent3", _I),
        ("aux", _F), ("aux2", _F), ("aux3", _F), ("aux4", _F),
        ("is_param", _I), ("no_decay", _I), ("frozen", _I),
    ]

def _tensor_struct(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(_NtTensor)).contents

def _entry_output_data(loss_or_logits_idx):
    """Read the output tensor of tape entry idx as an _NtTensor (data + len)."""
    tape = _lib.nt_tape_get()
    entry = ctypes.cast(tape, ctypes.POINTER(_NtTapeEntry))[loss_or_logits_idx]
    return _tensor_struct(entry.output)

# ═══════════════════════════════════════════════════════════════════════════════
# TENSOR / PARAMETER
# ═══════════════════════════════════════════════════════════════════════════════

class Tensor:
    def __init__(self, ptr, owns=True):
        self._ptr = ptr
        self._owns = owns

    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Tensor(ptr)

    @staticmethod
    def ones(size):
        t = Tensor.zeros(size)
        _lib.nt_tensor_fill(t._ptr, 1.0)
        return t

    @property
    def numel(self):
        return _tensor_struct(self._ptr).len

    @property
    def shape(self):
        s = _tensor_struct(self._ptr)
        return tuple(s.shape[i] for i in range(s.ndim))

    def fill_(self, val):
        _lib.nt_tensor_fill(self._ptr, _F(val)); return self

    def xavier_(self, fan_in, fan_out):
        _lib.nt_tensor_xavier(self._ptr, fan_in, fan_out); return self

    def rand_(self, scale):
        _lib.nt_tensor_rand(self._ptr, _F(scale)); return self

    def set_data(self, flat):
        s = _tensor_struct(self._ptr)
        n = min(len(flat), s.len)
        for i in range(n):
            s.data[i] = flat[i]

    def get_data(self):
        s = _tensor_struct(self._ptr)
        return [s.data[i] for i in range(s.len)]

    def __del__(self):
        if getattr(self, '_owns', False) and getattr(self, '_ptr', None):
            _lib.nt_tensor_free(self._ptr)


class Parameter(Tensor):
    """Trainable parameter — registered on the tape each step."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SYSTEM (torch.nn-compatible surface)
# ═══════════════════════════════════════════════════════════════════════════════

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module):
            self.__dict__.setdefault('_modules', {})[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def train(self, mode=True):
        self._training = mode
        _lib.nt_train_mode(1 if mode else 0)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter.zeros((out_features, in_features))
        self.weight.xavier_(in_features, out_features)  # bias-free (matches notorch seq_linear)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = Parameter.zeros((num_embeddings, embedding_dim))
        self.weight.rand_(0.02)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = Parameter.ones(dim)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL (inference-time helpers for generation; pure Python on logit lists)
# ═══════════════════════════════════════════════════════════════════════════════

def softmax(logits, temperature=1.0):
    if temperature != 1.0:
        logits = [x / temperature for x in logits]
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def cross_entropy(logits, target):
    mx = max(logits)
    lse = math.log(sum(math.exp(x - mx) for x in logits)) + mx
    return -(logits[target] - lse)

def multinomial(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if cum >= r:
            return i
    return len(probs) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# NOTORCH ENGINE — builds the PostGPT dual-attention graph on the notorch tape
# ═══════════════════════════════════════════════════════════════════════════════

class NotorchEngine:
    """
    Forward/backward/update for PostGPT through the notorch C tape.

    `params` is the flat, ordered parameter list:
        [wte, wpe,
         (rms1, c_wq, c_wk, c_wv, r_wr, r_wv, wo, rms2, mlp_up, mlp_down) * n_layers,
         norm_f]
    lm_head is TIED to wte (no separate parameter).
    """
    PER_LAYER = 10  # rms1, c_wq, c_wk, c_wv, r_wr, r_wv, wo, rms2, mlp_up, mlp_down

    def __init__(self, params, cfg, optimizer):
        self.params = params
        self.cfg = cfg
        self.optimizer = optimizer  # ChuckOptimizer — clips + steps the live tape

    def _build_forward(self, tok_ids, tgt_ids):
        """Build the tape graph. tgt_ids=None -> return logits_idx, else loss_idx."""
        c = self.cfg
        T = len(tok_ids); dim = c['dim']; hd = c['head_dim']
        n_c = c['n_content']; n_r = c['n_rrpram']; L = c['n_layers']; V = c['vocab']

        _lib.nt_tape_start()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in self.params]

        tok_t = Tensor.zeros(T); tok_t.set_data([float(x) for x in tok_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, _F(0)); tok_t._owns = False
        if tgt_ids is not None:
            tgt_t = Tensor.zeros(T); tgt_t.set_data([float(x) for x in tgt_ids])
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, _F(0)); tgt_t._owns = False

        pi = 0
        wte_i = tape_ids[pi]; pi += 1
        wpe_i = tape_ids[pi]; pi += 1

        # h[t] = wte[tok[t]] + wpe[t]
        h = _lib.nt_seq_embedding(wte_i, wpe_i, tok_idx, T, dim)

        for _ in range(L):
            rms1_i = tape_ids[pi]; pi += 1
            cwq_i  = tape_ids[pi]; pi += 1
            cwk_i  = tape_ids[pi]; pi += 1
            cwv_i  = tape_ids[pi]; pi += 1
            rwr_i  = tape_ids[pi]; pi += 1
            rwv_i  = tape_ids[pi]; pi += 1
            wo_i   = tape_ids[pi]; pi += 1
            rms2_i = tape_ids[pi]; pi += 1
            up_i   = tape_ids[pi]; pi += 1
            down_i = tape_ids[pi]; pi += 1

            xn = _lib.nt_seq_rmsnorm(h, rms1_i, T, dim)
            # Content attention: QK^T causal (scale 1/sqrt(head_dim) inside the op)
            q = _lib.nt_seq_linear(cwq_i, xn, T)
            k = _lib.nt_seq_linear(cwk_i, xn, T)
            v = _lib.nt_seq_linear(cwv_i, xn, T)
            c_out = _lib.nt_mh_causal_attention(q, k, v, T, hd)        # [T, n_c*hd]
            # RRPRAM attention: scores = x_i . Wr[h][:,j], causal, no scale
            rv = _lib.nt_seq_linear(rwv_i, xn, T)                      # [T, n_r*hd]
            r_out = _lib.nt_rrpram_attention(rwr_i, xn, rv, T, dim, n_r, hd)
            attn = _lib.nt_concat(c_out, r_out, T)                     # [T, (n_c+n_r)*hd]
            proj = _lib.nt_seq_linear(wo_i, attn, T)
            h = _lib.nt_add(h, proj)
            # MLP: down(relu(up(x)))
            xn2 = _lib.nt_seq_rmsnorm(h, rms2_i, T, dim)
            up_o = _lib.nt_seq_linear(up_i, xn2, T)
            relu_o = _lib.nt_relu(up_o)
            down_o = _lib.nt_seq_linear(down_i, relu_o, T)
            h = _lib.nt_add(h, down_o)

        normf_i = tape_ids[pi]; pi += 1
        hf = _lib.nt_seq_rmsnorm(h, normf_i, T, dim)
        logits = _lib.nt_seq_linear(wte_i, hf, T)                      # tied lm_head
        if tgt_ids is None:
            return logits
        return _lib.nt_seq_cross_entropy(logits, tgt_idx, T, V)

    def step(self, tok_ids, tgt_ids):
        """One forward+backward+Chuck update. Returns loss value."""
        loss_idx = self._build_forward(tok_ids, tgt_ids)
        loss_val = _entry_output_data(loss_idx).data[0]
        _lib.nt_tape_backward(loss_idx)
        self.optimizer.step(loss_val)   # Chuck: clip_grads + chuck_step on the live tape
        _lib.nt_tape_clear()
        return loss_val

    def logits_last(self, tok_ids):
        """Forward (no grad bookkeeping needed) -> logits at the final position."""
        T = len(tok_ids); V = self.cfg['vocab']
        logits_idx = self._build_forward(tok_ids, None)
        out = _entry_output_data(logits_idx)
        last = (T - 1) * V
        vals = [out.data[last + j] for j in range(V)]
        _lib.nt_tape_clear()
        return vals

    def save(self, path):
        n = len(self.params)
        arr = (_V * n)(*[p._ptr for p in self.params])
        _lib.nt_save(path.encode(), arr, n)


def seed(s):
    _lib.nt_seed(ctypes.c_uint64(s))


__all__ = [
    'Tensor', 'Parameter', 'Module', 'Linear', 'Embedding', 'RMSNorm',
    'NotorchEngine', 'softmax', 'cross_entropy', 'multinomial', 'seed', '_lib',
]
