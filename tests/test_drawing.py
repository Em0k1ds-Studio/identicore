from pathlib import Path
from typing import List

import cv2
import pytest
from cv2.typing import MatLike
from inspireface.modules.inspireface import FaceInformation

from identicore import IdenticoreSession
from identicore.types import DrawingOpts

DATASET_PATH: Path = Path(__file__).parent / 'dataset'
OUT_PATH: Path = Path(__file__).parent / 'out'

THRESHOLD: float = 0.65

pikachu_session = IdenticoreSession(model='Pikachu')
megatron_session = IdenticoreSession(model='Megatron')

DrawOpts = DrawingOpts(draw_boxes=True, draw_landmarks=True)


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
def test_drawing_single(session: IdenticoreSession, image_name: str) -> None:
    image_path: Path = DATASET_PATH / image_name
    assert image_path.exists(), 'Param: <DATASET_PATH / image_name> as `Path` is not exists.'

    image: MatLike = IdenticoreSession.load_image(image_path=image_path)
    faces: List[FaceInformation] = session.face_detection(
        image=image, draw_opts=DrawOpts, for_identification=False, threshold=THRESHOLD
    )

    assert len(faces), 'Not found any face.'

    out_image_path: Path = OUT_PATH / image_name
    cv2.imwrite(filename=str(out_image_path), img=image)

    assert out_image_path.exists(), 'After-draw image saving failed.'


@pytest.mark.parametrize(argnames='session', argvalues=(pikachu_session, megatron_session))
@pytest.mark.parametrize(
    argnames='image_name',
    argvalues=[
        'multiple0.jpg',
        'multiple1.jpg',
        'multiple2.jpg',
        'multiple3.jpg',
    ],
)
def test_drawing_multiple(session: IdenticoreSession, image_name: str) -> None:
    image_path: Path = DATASET_PATH / image_name
    assert image_path.exists(), 'Param: <DATASET_PATH / image_name> as `Path` is not exists.'

    image: MatLike = IdenticoreSession.load_image(image_path=image_path)
    faces: List[FaceInformation] = session.face_detection(
        image=image, draw_opts=DrawOpts, for_identification=False, threshold=THRESHOLD
    )

    assert len(faces), 'Not found any face.'

    out_image_path: Path = OUT_PATH / image_name
    cv2.imwrite(filename=str(out_image_path), img=image)

    assert out_image_path.exists(), 'After-draw image saving failed.'
