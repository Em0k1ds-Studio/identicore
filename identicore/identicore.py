"""Core package for face recognition and similarity comparison using InspireFace SDK."""

from typing import Any, List, Optional, Tuple

import cv2
import inspireface
from cv2.typing import MatLike
from inspireface import HF_ENABLE_FACE_RECOGNITION, FaceInformation, InspireFaceSession
from inspireface.modules.core.native import HF_ENABLE_DETECT_MODE_LANDMARK
from numpy import (
    array,
    clip,
    dtype,
    ndarray,
)

from identicore.exceptions import FeaturesExtractionFailed, MultipleFacesDetected
from identicore.types import DrawingOpts, FaceComparisonResult, FaceFeaturesArray, ImagePath, InspireModel

BASE_SCALE_FACTOR: float = 1000.0


class IdenticoreSession:
    """Base session for face recognition and similarity comparison using InspireFace SDK.

    Attributes:
        _inspire_session (`InspireFaceSession`): An instance of InspireFaceSession for face recognition operations.
    """

    _inspire_session: InspireFaceSession

    def __init__(self, model: InspireModel, inspire_flags: int = 0) -> None:
        """Initializes the IdenticoreSession with a specified InspireFace model.

        Args:
            model (`InspireModel`): The name of the InspireFace model to use (e.g., 'Pikachu' or 'Megatron').
            inspire_flags (`int`): Additional InspireFaceSession flags (e.g., *HF_ENABLE_QUALITY*).
        """
        inspireface.reload(model_name=model)
        self._inspire_session = InspireFaceSession(
            param=HF_ENABLE_FACE_RECOGNITION | HF_ENABLE_DETECT_MODE_LANDMARK | inspire_flags
        )

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
        draw_opts: DrawingOpts,
        for_identification: bool,
        threshold: float = 0.65,
    ) -> List[FaceInformation]:
        """Detects faces in an image with optional bounding box drawing.

        **Note**: For further face identification purposes it's recommended to set *threshold* at least to 0.7.

        Args:
            image (`MatLike`): The input image as a MatLike object.
            draw_opts (`DrawingOpts`): Drawing options for detected faces.
            for_identification (`bool`): If True, raises an exception if multiple faces are detected.
            threshold (`float`): Minimum detection confidence threshold (default: 0.65).

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

        for face in faces:
            if draw_opts.draw_boxes:
                self._draw_bounding_box(image=image, face=face, label=draw_opts.boxes_label)

            if draw_opts.draw_landmarks:
                self._draw_landmarks(image=image, face=face)

        return faces

    def face_comparison(
        self,
        first_face: Tuple[MatLike, FaceInformation],
        second_face: Tuple[MatLike, FaceInformation],
        threshold: float = 0.85,
    ) -> FaceComparisonResult:
        """Compares two faces for similarity based on extracted features.

        **Note**: It is assumed that the `FaceInformation` objects correspond to the provided images.

        Args:
            first_face (`Tuple[MatLike, FaceInformation]`): A tuple of (image, FaceInformation) for the first face.
            second_face (`Tuple[MatLike, FaceInformation]`): A tuple of (image, FaceInformation) for the second face.
            threshold (`float`): Minimum similarity threshold for a positive match (default: 0.85).

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
        return clip(inspireface.feature_comparison(feature1=n0, feature2=n1) + 0.25, a_min=0.0, a_max=1.0)

    @staticmethod
    def _draw_bounding_box(image: MatLike, face: FaceInformation, label: str) -> None:
        """Draws a bounding box and label (at the bottom of the bounding box) around a detected face on the image.

        Args:
            image (`MatLike`): The input image as a MatLike object where the box will be drawn.
            face (`FaceInformation`): FaceInformation object containing location and detection data.
            label (`str`): Text label to display near the bounding box.
        """
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

        # Find the bottom-left point for text placement
        bottom_pts: ndarray[Tuple[Any], dtype[Any]] = rect[rect[:, 1] == rect[:, 1].max()]
        left_bottom: Tuple[int, int] = bottom_pts[bottom_pts[:, 0].argmin()]

        # Calculate text background rectangle coordinates
        text_bg_width: int = text_width + int(10 * scale_factor)
        text_bg_height: int = text_height + int(5 * scale_factor)

        cv2.rectangle(
            img=image,
            pt1=(left_bottom[0], left_bottom[1]),
            pt2=(left_bottom[0] + text_bg_width, left_bottom[1] + text_bg_height),
            color=(0, 255, 0),
            thickness=-1,
        )

        cv2.putText(
            img=image,
            text=label,
            org=(
                left_bottom[0] + int(5 * scale_factor),
                left_bottom[1] + text_height + int(2 * scale_factor),
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    def _draw_landmarks(self, image: MatLike, face: FaceInformation) -> None:
        # Get image dimensions
        img_height, img_width = image.shape[:2]

        # Calculate scaling factors based on image size
        scale_factor: float = min(img_width, img_height) / BASE_SCALE_FACTOR

        landmarks: ndarray[Tuple[int, int], dtype[Any]] = self._inspire_session.get_face_dense_landmark(face)

        for landmark in landmarks:
            cv2.circle(
                img=image,
                center=tuple(map(int, landmark)),
                radius=max(0, int(2.5 * scale_factor)),
                color=(208, 224, 64),
                thickness=cv2.FILLED,
            )
