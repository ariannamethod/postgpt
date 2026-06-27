"""
ariannamethod — the notorch shim that evicts PyTorch from PostGPT's training loop.

C line:  notorch.c / notorch.h  (pure-C neural framework — tensors, tape autograd,
                                 RRPRAM, Chuck; the lineage PostGPT's metaweights began)
Python:  notorch_nn.py          (ctypes drop-in for the torch training path)
         chuck.py               (the self-aware optimizer, nt_tape_chuck_step)

no torch. no pip install torch. no 2.7 GB of your soul. the runtime was always
zero-dependency; now training is too.
"""

from .notorch_nn import (
    Tensor, Parameter, Module, Linear, Embedding, RMSNorm,
    NotorchEngine, softmax, cross_entropy, multinomial, seed,
)
from .chuck import ChuckOptimizer

__all__ = [
    'Tensor', 'Parameter', 'Module', 'Linear', 'Embedding', 'RMSNorm',
    'NotorchEngine', 'softmax', 'cross_entropy', 'multinomial', 'seed',
    'ChuckOptimizer',
]
