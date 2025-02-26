"""Identicore-specific exceptions."""


class IdenticoreException(Exception):
    """Base class for all Identicore exceptions."""

    def __init__(self, *args) -> None:
        super().__init__(*args)


class FeaturesExtractionFailed(IdenticoreException):
    """Face features extraction failed.

    Attributes:
        index (`int`): Index of image that failed on features extraction
    """

    index: int

    def __init__(self, index: int, *args) -> None:
        self.index = index
        super().__init__(*args)


class MultipleFacesDetected(IdenticoreException):
    """Multiple faces has been detected.

    Attributes:
        quantity (`int`): Quantity of faces detected
    """

    quantity: int

    def __init__(self, quantity: int, *args) -> None:
        self.quantity = quantity
        super().__init__(*args)
