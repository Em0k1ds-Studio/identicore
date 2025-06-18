from cv2.typing import MatLike as MatLike
from inspireface import FaceInformation as FaceInformation

from identicore.exceptions import (
    FeaturesExtractionFailed as FeaturesExtractionFailed,
)
from identicore.exceptions import (
    MultipleFacesDetected as MultipleFacesDetected,
)
from identicore.types import (
    DrawingOpts as DrawingOpts,
)
from identicore.types import (
    FaceComparisonResult as FaceComparisonResult,
)
from identicore.types import (
    FaceFeaturesArray as FaceFeaturesArray,
)
from identicore.types import (
    ImagePath as ImagePath,
)
from identicore.types import (
    InspireModel as InspireModel,
)

BASE_SCALE_FACTOR: float

class IdenticoreSession:
    def __init__(self, model: InspireModel, inspire_flags: int = 0) -> None: ...
    @staticmethod
    def load_image(image_path: ImagePath) -> MatLike: ...
    def face_detection(
        self, image: MatLike, draw_opts: DrawingOpts, for_identification: bool, threshold: float = 0.65
    ) -> list[FaceInformation]: ...
    def face_comparison(
        self,
        first_face: tuple[MatLike, FaceInformation],
        second_face: tuple[MatLike, FaceInformation],
        threshold: float = 0.85,
    ) -> FaceComparisonResult: ...
