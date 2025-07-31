"""
Stub-реализация «кусочка» fairseq, нужного TIFA / ModelScope.
"""

import sys, types

# ── создаём и регистрируем цепочку под-модулей ─────────────────────────
pkg_root = __name__                       # 'fairseq'
for sub in ("data", "data.audio"):
    mod = types.ModuleType(f"{pkg_root}.{sub}")
    sys.modules[f"{pkg_root}.{sub}"] = mod

ft_mod = types.ModuleType(f"{pkg_root}.data.audio.feature_transforms")
sys.modules[f"{pkg_root}.data.audio.feature_transforms"] = ft_mod

# ── минимальные заглушки, которые могут запрашиваться ──────────────────
class _Dummy:                             # базовый «пустой» класс
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return None

ft_mod.AudioFeatureTransform       = _Dummy          # уже был нужен
ft_mod.CompositeAudioFeatureTransform = _Dummy      # 🔥 новый импорт
ft_mod.__all__ = [
    "AudioFeatureTransform",
    "CompositeAudioFeatureTransform",
]

# ─────────────────────────────────────────────────────────────────────
#  Stub для   fairseq.data.audio.speech_to_text_dataset.S2TDataConfig
# ─────────────────────────────────────────────────────────────────────
import types, sys, pathlib

root = __name__                          # 'fairseq'
pkg_path = pathlib.Path(__file__).parent

# гарантируем, что fairseq.data.audio уже есть (создавали ранее)
audio_pkg = sys.modules[f"{root}.data.audio"]

# создаём под-модуль
s2t_mod_name = f"{root}.data.audio.speech_to_text_dataset"
s2t_mod = types.ModuleType(s2t_mod_name)
sys.modules[s2t_mod_name] = s2t_mod

class S2TDataConfig:                     # пустая заглушка
    def __init__(self, *a, **kw): pass

s2t_mod.S2TDataConfig = S2TDataConfig
s2t_mod.__all__ = ["S2TDataConfig"]
# ─────────────────────────────────────────────────────────────────────
