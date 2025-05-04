from typing import List, Tuple

from cv2.typing import MatLike as MatLike
from inspireface import FaceInformation as FaceInformation
from numpy import dtype as dtype
from numpy import ndarray as ndarray
from numpy import signedinteger as signedinteger

from identicore.exceptions import FeaturesExtractionFailed as FeaturesExtractionFailed
from identicore.exceptions import MultipleFacesDetected as MultipleFacesDetected
from identicore.types import FaceComparisonResult as FaceComparisonResult
from identicore.types import FaceFeaturesArray as FaceFeaturesArray
from identicore.types import ImagePath as ImagePath
from identicore.types import InspireModel as InspireModel

class IdenticoreSession:
    def __init__(self, model: InspireModel, inspire_flags: int = 0) -> None: ...
    @staticmethod
    def load_image(image_path: ImagePath) -> MatLike: ...
    def face_detection(
        self,
        image: MatLike,
        for_identification: bool,
        threshold: float = 0.65,
        draw_boxes: bool = False,
        boxes_label: str = 'human',
    ) -> List[FaceInformation]: ...
    def face_comparison(
        self,
        first_face: Tuple[MatLike, FaceInformation],
        second_face: Tuple[MatLike, FaceInformation],
        threshold: float = 0.75,
    ) -> FaceComparisonResult: ...
