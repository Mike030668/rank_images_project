"""
Минимальный пакет fairseq.data – достаточно для Ofa/MMSpeech.
Экспортируем только compute_mask_indices.
"""
from .data_utils import compute_mask_indices

__all__ = ["compute_mask_indices"]

"""
Minimal stub so that `fairseq.data.data_utils` resolves.

We just re-export everything from the top-level `fairseq.data_utils`.
"""
from ..data_utils import *          # noqa: F401,F403
