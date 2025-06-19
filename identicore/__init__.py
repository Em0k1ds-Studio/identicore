"""Identicore: A face recognition and similarity comparison library using InspireFace SDK."""

from .__version__ import __version__
from .exceptions import (
    FeaturesExtractionFailed,
    IdenticoreException,
    MultipleFacesDetected,
)
from .identicore import IdenticoreSession

# if TYPE_CHECKING:
from .types import (
    DefaultDrawingOpts,
    DrawingOpts,
    FaceComparisonResult,
    FaceFeaturesArray,
    ImagePath,
    InspireModel,
)

__all__ = (
    '__version__',
    'IdenticoreException',
    'FeaturesExtractionFailed',
    'MultipleFacesDetected',
    'IdenticoreSession',
    'FaceComparisonResult',
    'ImagePath',
    'InspireModel',
    'FaceFeaturesArray',
    'DefaultDrawingOpts',
    'DrawingOpts',
)
