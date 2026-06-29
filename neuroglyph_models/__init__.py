"""Neural decoders for EPOC X keystroke-aligned EEG."""

from neuroglyph_models.tiny_b2q import TinyB2Q
from neuroglyph_models.eegnet import EEGNet
from neuroglyph_models.heads import TASK_HEADS

__all__ = ["TinyB2Q", "EEGNet", "TASK_HEADS"]