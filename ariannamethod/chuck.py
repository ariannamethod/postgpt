"""
chuck.py — the Chuck optimizer via notorch (nt_tape_chuck_step). No torch.optim.

PostGPT's training loop used to carry a hand-written AdamW-with-self-awareness
re-implementation of Chuck on top of torch.optim. This is the real thing: the
C Chuck (`nt_tape_chuck_step`) with loss-aware damping, per-parameter gradient
monitoring, stagnation noise, parameter freezing and macro-patience — the same
optimizer that drove the Arianna LoRA SFT at production scale.

θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η

Adam is blind. Chuck sees. Chuck remembers.
"""

from .notorch_nn import _lib
import ctypes


class ChuckOptimizer:
    """
    Self-aware optimizer backed by notorch's nt_tape_chuck_step.

    Used by NotorchEngine between backward and tape-clear:
        loss = engine.step(tok, tgt)   # engine calls optimizer.step(loss) internally
    or directly on a live tape:
        nt_tape_backward(loss_idx); optimizer.step(loss_val); nt_tape_clear()
    """

    def __init__(self, lr=3e-4, max_grad_norm=1.0):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def step(self, loss_val):
        """Clip gradients, then run one Chuck step on the active tape."""
        self.global_step += 1
        _lib.nt_tape_clip_grads(ctypes.c_float(self.max_grad_norm))
        _lib.nt_tape_chuck_step(ctypes.c_float(self.lr), ctypes.c_float(loss_val))

    def zero_grad(self):
        """No-op — the notorch tape is reset by nt_tape_clear() each step."""
        pass
