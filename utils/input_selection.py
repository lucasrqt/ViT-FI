import enum


class InputSelectionMethod(enum.Enum):
    """
    Enum for input selection methods.
    """

    DSA = 0
    MAX_P = 1
    VARIANCE = 2
    WEIGHTED_VARIANCE = 3
    CONFIDENCE = 4

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
