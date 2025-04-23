from methods import InputSelection, InputSelectionFactory
from utils.input_selection import InputSelectionMethod


@InputSelectionFactory.register(InputSelectionMethod.CONFIDENCE)
class Confidence(InputSelection):
    """
    Confidence input selection method.
    """

    def __init__(self):
        super().__init__(method=InputSelectionMethod.CONFIDENCE)

    def select_input(self):
        raise NotImplementedError("Not implemented yet.")
