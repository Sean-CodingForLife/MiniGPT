"""Pure-Python backend (Stage 2): self-written tensors, autograd and GPT.

The package intentionally depends on the standard library only:

* :mod:`minigpt.pure.tensor` — dense tensor + reverse-mode autograd
* :mod:`minigpt.pure.adamw` — AdamW + gradient clipping
* :mod:`minigpt.pure.gpt` — GPT forward/generate with the same architecture
* :mod:`minigpt.pure.weights` — torch-free weight file (float32 + JSON index)

Higher layers are imported explicitly so the minimal inference bundle stays
small and torch-free:

* :mod:`minigpt.pure.backend` — ``PureBackend`` (registers ``BACKENDS["pure"]``)
* :mod:`minigpt.pure.checkpoint` — ``.mgckpt`` checkpoint format
* :mod:`minigpt.pure.inference` — load a ``.mgckpt`` bundle for chat/evaluate
"""

from minigpt.pure.adamw import AdamW, clip_grad_norm
from minigpt.pure.gpt import PureGPT, PureGPTConfig
from minigpt.pure.tensor import Tensor, no_grad
from minigpt.pure.weights import WeightFileError, load_weights, save_weights

__all__ = [
    "Tensor",
    "no_grad",
    "AdamW",
    "clip_grad_norm",
    "PureGPT",
    "PureGPTConfig",
    "load_weights",
    "save_weights",
    "WeightFileError",
]
