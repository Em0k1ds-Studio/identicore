"""Identicore-related types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from numpy import dtype, floating, ndarray
from numpy._typing._nbit_base import _32Bit

InspireModel = Literal['Pikachu', 'Megatron']
ImagePath = Union[Path, str]
FaceFeaturesArray = ndarray[tuple[int], dtype[floating[_32Bit]]]


@dataclass
class FaceComparisonResult:
    """Result of a face comparison operation.

    Attributes:
        is_match (`bool`): True if the faces match based on the similarity threshold.
        similarity_confidence (`float`): The computed similarity score between 0 and 1.
    """

    is_match: bool
    similarity_confidence: float


@dataclass
class DrawingOpts:
    """Drawing options for detected faces.

    Attributes:
        draw_boxes (`bool`): Draws bounding boxes around detected faces (default: False)
        boxes_label (`str`): Label text for bounding boxes (default: 'human')
        draw_landmarks (`bool`): Draw face dense landmarks on detected faces (default: False)
    """

    draw_boxes: bool = False
    boxes_label: str = 'human'

    draw_landmarks: bool = False


DefaultDrawingOpts = DrawingOpts()
