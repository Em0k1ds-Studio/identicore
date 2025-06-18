from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from numpy import dtype, floating, ndarray
from numpy._typing._nbit_base import _32Bit

InspireModel: Literal['Pikachu', 'Megatron']
ImagePath = Path | str
FaceFeaturesArray: ndarray[tuple[int], dtype[floating[_32Bit]]]

@dataclass
class FaceComparisonResult:
    is_match: bool
    similarity_confidence: float

@dataclass
class DrawingOpts:
    draw_boxes: bool = ...
    boxes_label: str = ...
    draw_landmarks: bool = ...

DefaultDrawingOpts: DrawingOpts
