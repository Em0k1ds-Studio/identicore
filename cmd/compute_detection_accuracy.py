"""Compute detection accuracy using dataset from 'thispersondoesnotexist'."""

import asyncio

# import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from hashlib import md5

# from pathlib import Path
from typing import Any, Coroutine, Deque, List, Set

import aiohttp
import cv2
from cv2.typing import MatLike
from inspireface.modules.inspireface import FaceInformation
from numpy import frombuffer, uint8

from identicore.identicore import IdenticoreSession
from identicore.types import DefaultDrawingOpts

THRESHOLD: float = 0.65

SAMPLES: int = 1000
WORKERS: int = 10

DELAY: float = 0.1
DUPLICATE_RETRIES: int = 5

# WORK_DIR: Path = Path(__file__).parent / 'out'

identicore_session = IdenticoreSession(model='Megatron')
fetched_digests: Set[str] = set()


@dataclass
class SampleResult:
    is_match: bool
    detection_confidence: float


async def worker(session: aiohttp.ClientSession, queue: Deque[SampleResult], id: int, sample_size: int) -> None:
    def bytes2cvimg(x: bytes) -> MatLike:
        return cv2.imdecode(buf=frombuffer(buffer=x, dtype=uint8), flags=cv2.IMREAD_COLOR)

    for _ in range(sample_size):
        for _ in range(DUPLICATE_RETRIES):
            async with session.get(url='https://thispersondoesnotexist.com/', allow_redirects=False) as response:
                response_raw: bytes = await response.read()

            digest: str = md5(string=response_raw).hexdigest()

            if digest in fetched_digests:
                await asyncio.sleep(delay=DELAY)
                continue

            fetched_digests.add(digest)
            response_image: MatLike = bytes2cvimg(x=response_raw)

            faces: List[FaceInformation] = identicore_session.face_detection(
                image=response_image, draw_opts=DefaultDrawingOpts, for_identification=False, threshold=0.00
            )
            assert len(faces) == 1

            face: FaceInformation = faces[0]

            queue.append(
                SampleResult(
                    is_match=face.detection_confidence >= THRESHOLD, detection_confidence=face.detection_confidence
                )
            )

            print(
                f' :: \x1b[2m<Coro-{id:02d}>\x1b[0m {digest} -> \x1b[1m{(face.detection_confidence * 100):.04f}%\x1b[0m'
            )
            break

        await asyncio.sleep(delay=DELAY)


async def main(_argc: int, _argv: list[str]) -> int:
    # if not WORK_DIR.exists():
    #     os.mkdir(path=WORK_DIR)

    sample_size: int = SAMPLES // WORKERS

    print('\n .. \x1b[1mcompute-detection-accuracy\x1b[0m')
    print(f' :: sample_size: {sample_size}, workers_count: {WORKERS}')

    queue: Deque[SampleResult] = deque()
    aiohttp_session = aiohttp.ClientSession(
        headers={
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',  # noqa: E501
        }
    )
    entry_time: float = time.monotonic()

    workers: List[Coroutine[Any, Any, None]] = [
        worker(session=aiohttp_session, queue=queue, id=i + 1, sample_size=sample_size) for i in range(WORKERS)
    ]
    await asyncio.gather(*workers)

    elapsed_time: float = time.monotonic() - entry_time
    await aiohttp_session.close()

    results: List[SampleResult] = list(queue)

    print(f' :: elapsed: \x1b[30;47m{elapsed_time:.3f}s\x1b[0m')
    print(
        f'\n\x1b[32m{"compute-detection-accuracy":<30}\x1b[0m{"accuracy:":<12}'
        + f'\x1b[1m{(len(list(filter(lambda x: x.is_match, results))) / len(results)) * 100:.04f}%\x1b[0m\n'
        + f'{"":<30}{"confidence:":<12}['
        + f'\x1b[2m{min(results, key=lambda x: x.detection_confidence).detection_confidence * 100:.04f}%\x1b[0m '
        + f'\x1b[1m{sum([x.detection_confidence for x in results]) / len(results) * 100:.04f}%\x1b[0m '
        + f'\x1b[2m{max(results, key=lambda x: x.detection_confidence).detection_confidence * 100:.04f}%\x1b[0m]\n'
    )

    return 0


if __name__ == '__main__':
    try:
        status: int = asyncio.run(main=main(_argc=len(sys.argv), _argv=sys.argv))
    except KeyboardInterrupt:
        status = 0

    sys.exit(status)
