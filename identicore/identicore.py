"""Core package for face recognition and similarity comparison using InspireFace SDK."""

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import cv2
import inspireface
from cv2.typing import MatLike
from inspireface import HF_ENABLE_FACE_RECOGNITION, FaceInformation, InspireFaceSession
from numpy import (
    argmax,
    array,
    dot,
    dtype,
    floating,
    ndarray,
    signedinteger,
)
from numpy._typing._nbit_base import _32Bit

from identicore.exceptions import FeaturesExtractionFailed, MultipleFacesDetected

type InspireModel = Literal['Pikachu', 'Megatron']
type ImagePath = Union[Path, str]
type FeaturesArray = ndarray[tuple[int], dtype[floating[_32Bit]]]


class IdenticoreSession:
    """Base session for face recognition and similarity comparison using InspireFace SDK."""

    _inspire_session: InspireFaceSession

    def __init__(self, model: InspireModel) -> None:
        """Sample."""
        inspireface.reload(model_name=model)
        self._inspire_session = InspireFaceSession(param=HF_ENABLE_FACE_RECOGNITION)

    @staticmethod
    def load_image(image_path: ImagePath) -> MatLike:
        """Sample."""
        return cv2.imread(filename=str(image_path))

    def face_detection(
        self,
        image: MatLike,
        for_identification: bool,
        threshold: float = 0.5,
        draw_boxes: bool = True,
        boxes_label: str = 'human',
    ) -> List[FaceInformation]:
        """Sample."""
        faces: List[FaceInformation] = list(
            filter(lambda x: x.detection_confidence >= threshold, self._inspire_session.face_detection(image=image))
        )

        if for_identification and len(faces) > 1:
            raise MultipleFacesDetected(quantity=len(faces))

        if draw_boxes:
            for face in faces:
                self._draw_bounding_box(image, face, boxes_label)

        return faces

    def face_comparison(
        self,
        first_face: Tuple[MatLike, FaceInformation],
        second_face: Tuple[MatLike, FaceInformation],
        threshold: float = 0.7,
    ) -> Tuple[bool, float]:
        """Sample. Returns (bool, similarity_confidence)."""
        first_features: Optional[FeaturesArray] = self._inspire_session.face_feature_extract(
            image=first_face[0], face_information=first_face[1]
        )

        if first_features is None:
            raise FeaturesExtractionFailed(index=0)

        second_features: Optional[FeaturesArray] = self._inspire_session.face_feature_extract(
            image=second_face[0], face_information=second_face[1]
        )

        if second_features is None:
            raise FeaturesExtractionFailed(index=1)

        similarity_confidence: float = self._cosine_similarity(first_features, second_features)

        return (similarity_confidence >= threshold, similarity_confidence)

    @staticmethod
    def _cosine_similarity(n0: FeaturesArray, n1: FeaturesArray) -> float:
        """Cosine similarity, but it's result belongs to the interval [0;1]."""
        return (dot(a=n0, b=n1) / (cv2.norm(src1=n0) * cv2.norm(src1=n1)) + 1) / 2

    @staticmethod
    def _draw_bounding_box(image: MatLike, face: FaceInformation, label: str) -> None:
        x1, y1, x2, y2 = face.location

        # Construct a rectangle from given upper-left & bottom-right points
        rect: ndarray[Tuple[int, ...], dtype[Any]] = array(
            object=[
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ],
            dtype=int,
        )

        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Calculate scaling factors based on image size
        scale_factor: float = min(img_width, img_height) / 1000.0

        box_thickness: int = max(2, int(2 * scale_factor))
        text_thickness: int = max(1, int(1.3 * scale_factor))
        font_scale: float = max(0.5, 0.7 * scale_factor)

        cv2.polylines(
            img=image,
            pts=[rect],
            isClosed=True,
            color=(0, 255, 0),
            thickness=box_thickness,
        )

        (text_width, text_height), _ = cv2.getTextSize(
            text=label,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            thickness=text_thickness,
        )

        # Find the bottom-most point for text placement
        bottom_idx: signedinteger = argmax(a=rect[:, 1])
        bottom_point: Tuple[int, int] = rect[bottom_idx]

        # Calculate text background rectangle coordinates
        text_bg_width: int = text_width + int(10 * scale_factor)
        text_bg_height: int = text_height + int(5 * scale_factor)

        cv2.rectangle(
            img=image,
            pt1=(bottom_point[0], bottom_point[1]),
            pt2=(bottom_point[0] + text_bg_width, bottom_point[1] + text_bg_height),
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.putText(
            img=image,
            text=label,
            org=(
                bottom_point[0] + int(5 * scale_factor),
                bottom_point[1] + text_height + int(2 * scale_factor),
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )
