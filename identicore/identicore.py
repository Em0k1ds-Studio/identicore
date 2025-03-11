"""Core package for face recognition and similarity comparison using InspireFace SDK."""

from typing import Any, List, Optional, Tuple

import cv2
import inspireface
from cv2.typing import MatLike
from inspireface import HF_ENABLE_FACE_RECOGNITION, FaceInformation, InspireFaceSession
from numpy import (
    argmax,
    array,
    dot,
    dtype,
    ndarray,
    signedinteger,
)

from identicore.exceptions import FeaturesExtractionFailed, MultipleFacesDetected
from identicore.types import FaceComparisonResult, FaceFeaturesArray, ImagePath, InspireModel


class IdenticoreSession:
    """Base session for face recognition and similarity comparison using InspireFace SDK.

    Attributes:
        _inspire_session (`InspireFaceSession`): An instance of InspireFaceSession for face recognition operations.
    """

    _inspire_session: InspireFaceSession

    def __init__(self, model: InspireModel) -> None:
        """Initializes the IdenticoreSession with a specified InspireFace model.

        Args:
            model (`InspireModel`): The name of the InspireFace model to use (e.g., 'Pikachu' or 'Megatron').
        """
        inspireface.reload(model_name=model)
        self._inspire_session = InspireFaceSession(param=HF_ENABLE_FACE_RECOGNITION)

    @staticmethod
    def load_image(image_path: ImagePath) -> MatLike:
        """Loads an image from the specified path.

        Args:
            image_path (`ImagePath`): The path to the image file, either as a Path object or string.

        Returns:
            `MatLike`: A MatLike object representing the loaded image in OpenCV format.
        """
        return cv2.imread(filename=str(image_path))

    def face_detection(
        self,
        image: MatLike,
        for_identification: bool,
        threshold: float = 0.65,
        draw_boxes: bool = False,
        boxes_label: str = 'human',
    ) -> List[FaceInformation]:
        """Detects faces in an image with optional bounding box drawing.

        **Note**: For further face identification purposes it's recommended to set *threshold* at least to 0.7.

        Args:
            image (`MatLike`): The input image as a MatLike object.
            for_identification (`bool`): If True, raises an exception if multiple faces are detected.
            threshold (`float`): Minimum detection confidence threshold (default: 0.65).
            draw_boxes (`bool`): If True, draws bounding boxes around detected faces (default: False).
            boxes_label (`str`): Label text for bounding boxes (default: 'human').

        Returns:
            `List[FaceInformation]`: A list of FaceInformation objects for detected faces.

        Raises:
            `MultipleFacesDetected`: If *for_identification* is True and multiple faces are detected.
        """
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
        threshold: float = 0.75,
    ) -> FaceComparisonResult:
        """Compares two faces for similarity based on extracted features.

        **Note**: It is assumed that the `FaceInformation` objects correspond to the provided images.

        Args:
            first_face (`Tuple[MatLike, FaceInformation]`): A tuple of (image, FaceInformation) for the first face.
            second_face (`Tuple[MatLike, FaceInformation]`): A tuple of (image, FaceInformation) for the second face.
            threshold (`float`): Minimum similarity threshold for a positive match (default: 0.75).

        Returns:
            `FaceComparisonResult`: An object containing
                - is_match (`bool`): True if similarity exceeds threshold, False otherwise.
                - similarity_confidence (`float`): The computed similarity score between 0 and 1.

        Raises:
            `FeaturesExtractionFailed`: If feature extraction fails for either face.
        """
        first_image, first_face_info = first_face
        second_image, second_face_info = second_face

        first_features: Optional[FaceFeaturesArray] = self._inspire_session.face_feature_extract(
            image=first_image, face_information=first_face_info
        )

        if first_features is None:
            raise FeaturesExtractionFailed(0, 'Feature extraction failed for the first face.')

        second_features: Optional[FaceFeaturesArray] = self._inspire_session.face_feature_extract(
            image=second_image, face_information=second_face_info
        )

        if second_features is None:
            raise FeaturesExtractionFailed(1, 'Feature extraction failed for the second face.')

        similarity_confidence: float = self._cosine_similarity(first_features, second_features)

        return FaceComparisonResult(
            is_match=similarity_confidence >= threshold, similarity_confidence=similarity_confidence
        )

    @staticmethod
    def _cosine_similarity(n0: FaceFeaturesArray, n1: FaceFeaturesArray) -> float:
        """Computes the cosine similarity between two face-feature arrays, normalized to [0, 1].

        Args:
            n0 (`FaceFeaturesArray`): First face-feature array.
            n1 (`FaceFeaturesArray`): Second face-feature array.

        Returns:
            `float`: The cosine similarity score between 0 and 1.
        """
        return (dot(a=n0, b=n1) / (cv2.norm(src1=n0) * cv2.norm(src1=n1)) + 1) / 2

    @staticmethod
    def _draw_bounding_box(image: MatLike, face: FaceInformation, label: str) -> None:
        """Draws a bounding box and label (at the bottom of the bounding box) around a detected face on the image.

        Args:
            image (`MatLike`): The input image as a MatLike object where the box will be drawn.
            face (`FaceInformation`): FaceInformation object containing location and detection data.
            label (`str`): Text label to display near the bounding box.
        """
        BASE_SCALE_FACTOR: float = 1000.0
        MIN_BOX_THICKNESS: int = 2
        MIN_TEXT_THICKNESS: int = 1
        MIN_FONT_SCALE: float = 0.5

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
        scale_factor: float = min(img_width, img_height) / BASE_SCALE_FACTOR
        box_thickness: int = max(MIN_BOX_THICKNESS, int(2 * scale_factor))
        text_thickness: int = max(MIN_TEXT_THICKNESS, int(1.3 * scale_factor))
        font_scale: float = max(MIN_FONT_SCALE, 0.7 * scale_factor)

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
