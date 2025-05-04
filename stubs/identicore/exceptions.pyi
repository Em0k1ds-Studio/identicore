class IdenticoreException(Exception):
    def __init__(self, *args) -> None: ...

class FeaturesExtractionFailed(IdenticoreException):
    index: int
    def __init__(self, index: int, *args) -> None: ...

class MultipleFacesDetected(IdenticoreException):
    quantity: int
    def __init__(self, quantity: int, *args) -> None: ...
