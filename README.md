# identicore
> Core package for face recognition and similarity comparison using [InspireFace SDK](https://github.com/HyperInspire/InspireFace)

## Installation
```shell
pip install --upgrade git+https://github.com/Em0k1ds-Studio/identicore@dev
```

## Usage
```python
from identicore import IdenticoreSession

# Initialize session with Pikachu model
session = IdenticoreSession(model='Pikachu')

# Load and detect faces in an images
first_image = session.load_image('path/to/first_image.jpg')
first_faces = session.face_detection(first_image, for_identification=True)

second_image = session.load_image('path/to/second_image.png')
second_faces = session.face_detection(second_image, for_identification=True)

# Compare two faces
result = session.face_comparison(
    first_face=(first_image, first_faces[0]),
    second_face=(second_image, second_faces[0]),
)

print(f'Match: {result.is_match}, Confidence: {result.similarity_confidence}')
```

## License
`identicore` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
