from pathlib import Path
from typing import List

import pytest
from cv2.typing import MatLike
from inspireface.modules.inspireface import FaceInformation

from identicore import IdenticoreSession
from identicore.types import FaceComparisonResult

DATASET_PATH: Path = Path(__file__).parent / 'dataset'

IDENTIFICATION_THRESHOLD: float = 0.7
COMPARISON_THRESHOLD: float = 0.75

pikachu_session = IdenticoreSession(model='Pikachu')
megatron_session = IdenticoreSession(model='Megatron')


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='first_image_name, second_image_name',
    argvalues=(
        ('elon0.jpg', 'elon1.jpg'),
        ('snowden0.jpg', 'snowden1.jpg'),
    ),
)
def test_positive_comparison(session: IdenticoreSession, first_image_name: str, second_image_name: str) -> None:
    first_image_path: Path = DATASET_PATH / first_image_name
    second_image_path: Path = DATASET_PATH / second_image_name

    assert first_image_path.exists(), 'Param: <DATASET_PATH / first_image_path> as `Path` is not exists.'
    assert second_image_path.exists(), 'Param: <DATASET_PATH / second_image_path> as `Path` is not exists.'

    first_image: MatLike = IdenticoreSession.load_image(image_path=first_image_path)
    first_faces: List[FaceInformation] = session.face_detection(
        image=first_image, for_identification=False, threshold=IDENTIFICATION_THRESHOLD
    )

    second_image: MatLike = IdenticoreSession.load_image(image_path=second_image_path)
    second_faces: List[FaceInformation] = session.face_detection(
        image=second_image, for_identification=False, threshold=IDENTIFICATION_THRESHOLD
    )

    assert len(first_faces) == 1, f'Found {len(first_faces)}, but expected 1.'
    assert first_faces[0].detection_confidence >= IDENTIFICATION_THRESHOLD, (
        f'Got detection confidence on <first_image> less than expected: {first_faces[0].detection_confidence}.'
    )

    assert len(second_faces) == 1, f'Found {len(second_faces)}, but expected 1.'
    assert second_faces[0].detection_confidence >= IDENTIFICATION_THRESHOLD, (
        f'Got detection confidence on <second_image> less than expected: {second_faces[0].detection_confidence}.'
    )

    similarity: FaceComparisonResult = session.face_comparison(
        first_face=(first_image, first_faces[0]),
        second_face=(second_image, second_faces[0]),
        threshold=COMPARISON_THRESHOLD,
    )

    assert similarity.is_match, 'Face comparison not matched.'
    assert similarity.similarity_confidence >= COMPARISON_THRESHOLD, (
        f'Got similarity confidence less than expected: {similarity.similarity_confidence}.'
    )


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='first_image_name, second_image_name',
    argvalues=(
        ('elon0.jpg', 'snowden0.jpg'),
        ('snowden0.jpg', 'elon0.jpg'),
        ('elon1.jpg', 'snowden0.jpg'),
        ('elon0.jpg', 'snowden1.jpg'),
    ),
)
def test_negative_comparison(session: IdenticoreSession, first_image_name: str, second_image_name: str) -> None:
    first_image_path: Path = DATASET_PATH / first_image_name
    second_image_path: Path = DATASET_PATH / second_image_name

    assert first_image_path.exists(), 'Param: <DATASET_PATH / first_image_path> as `Path` is not exists.'
    assert second_image_path.exists(), 'Param: <DATASET_PATH / second_image_path> as `Path` is not exists.'

    first_image: MatLike = IdenticoreSession.load_image(image_path=first_image_path)
    first_faces: List[FaceInformation] = session.face_detection(
        image=first_image, for_identification=False, threshold=IDENTIFICATION_THRESHOLD
    )

    second_image: MatLike = IdenticoreSession.load_image(image_path=second_image_path)
    second_faces: List[FaceInformation] = session.face_detection(
        image=second_image, for_identification=False, threshold=IDENTIFICATION_THRESHOLD
    )

    assert len(first_faces) == 1, f'Found {len(first_faces)}, but expected 1.'
    assert first_faces[0].detection_confidence >= IDENTIFICATION_THRESHOLD, (
        f'Got detection confidence on <first_image> less than expected: {first_faces[0].detection_confidence}.'
    )

    assert len(second_faces) == 1, f'Found {len(second_faces)}, but expected 1.'
    assert second_faces[0].detection_confidence >= IDENTIFICATION_THRESHOLD, (
        f'Got detection confidence on <second_image> less than expected: {second_faces[0].detection_confidence}.'
    )

    similarity: FaceComparisonResult = session.face_comparison(
        first_face=(first_image, first_faces[0]),
        second_face=(second_image, second_faces[0]),
        threshold=COMPARISON_THRESHOLD,
    )

    assert not similarity.is_match, 'Face comparison matched.'
    assert similarity.similarity_confidence < COMPARISON_THRESHOLD, (
        f'Got similarity confidence more or equal than expected: {similarity.similarity_confidence}.'
    )
