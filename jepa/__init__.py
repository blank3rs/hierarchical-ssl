"""
JEPA for text - learns representations by predicting masked spans.

ContextEncoder: processes visible text (trainable)
TargetEncoder: processes masked text (EMA-updated)
Predictor: maps context to target predictions (trainable)
"""

from jepa.context import ContextEncoder
from jepa.target import TargetEncoder
from jepa.predictor import Predictor
from jepa.model import JEPA

__all__ = [
    'ContextEncoder',
    'TargetEncoder',
    'Predictor',
    'JEPA',
]

__version__ = '0.1.0'

