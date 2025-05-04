from dataclasses import dataclass
from pathlib import Path

from _typeshed import Incomplete

InspireModel: Incomplete
ImagePath = Path | str
FaceFeaturesArray: Incomplete

@dataclass
class FaceComparisonResult:
    is_match: bool
    similarity_confidence: float
    def __init__(self, is_match, similarity_confidence) -> None: ...
