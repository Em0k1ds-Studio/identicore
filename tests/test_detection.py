from pathlib import Path
from typing import List

import pytest
from cv2.typing import MatLike
from inspireface.modules.inspireface import FaceInformation

from identicore import IdenticoreSession
from identicore.types import DefaultDrawingOpts

DATASET_PATH: Path = Path(__file__).parent / 'dataset'

THRESHOLD: float = 0.65
IDENTIFICATION_THRESHOLD: float = 0.7

pikachu_session = IdenticoreSession(model='Pikachu')
megatron_session = IdenticoreSession(model='Megatron')


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='image_name',
    argvalues=(
        'single0.jpg',
        'single1.jpg',
        'single2.jpg',
        'single3.jpg',
    ),
)
def test_positive_detection_single(session: IdenticoreSession, image_name: str) -> None:
    image_path: Path = DATASET_PATH / image_name
    assert image_path.exists(), 'Param: <DATASET_PATH / image_name> as `Path` is not exists.'

    image: MatLike = IdenticoreSession.load_image(image_path=image_path)
    faces: List[FaceInformation] = session.face_detection(
        image=image, draw_opts=DefaultDrawingOpts, for_identification=False, threshold=THRESHOLD
    )

    assert len(faces) == 1, f'Found {len(faces)}, but expected 1.'
    assert faces[0].detection_confidence >= THRESHOLD, (
        f'Got detection confidence less than expected: {faces[0].detection_confidence}.'
    )


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='image_name, expected_faces',
    argvalues=[
        ('multiple0.jpg', 5),
        ('multiple1.jpg', 8),
        ('multiple2.jpg', 5),
        ('multiple3.jpg', 5),
    ],
)
def test_positive_detection_multiple(session: IdenticoreSession, image_name: str, expected_faces: int) -> None:
    image_path: Path = DATASET_PATH / image_name
    assert image_path.exists(), 'Param: <DATASET_PATH / image_name> as `Path` is not exists.'

    image: MatLike = IdenticoreSession.load_image(image_path=image_path)
    faces: List[FaceInformation] = session.face_detection(
        image=image, draw_opts=DefaultDrawingOpts, for_identification=False, threshold=THRESHOLD
    )

    assert len(faces) == expected_faces, f'Found {len(faces)}, but expected {expected_faces}.'
    assert faces[0].detection_confidence >= THRESHOLD, (
        f'Got detection confidence less than expected: {faces[0].detection_confidence}.'
    )


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='image_name',
    argvalues=(
        'cat.jpg',
        'dog.jpg',
        'monkey.jpg',
        'hide.jpg',
    ),
)
def test_negative_detection(session: IdenticoreSession, image_name: str) -> None:
    image_path: Path = DATASET_PATH / image_name
    assert image_path.exists(), 'Param: <DATASET_PATH / image_name> as `Path` is not exists.'

    image: MatLike = IdenticoreSession.load_image(image_path=image_path)
    faces: List[FaceInformation] = session.face_detection(
        image=image,
        draw_opts=DefaultDrawingOpts,
        for_identification=False,
        threshold=IDENTIFICATION_THRESHOLD,
    )

    assert not len(faces), f'Found {len(faces)}, but expected 0.'
