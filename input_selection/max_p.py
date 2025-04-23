from methods import InputSelection, InputSelectionFactory
from utils.input_selection import InputSelectionMethod


@InputSelectionFactory.register(InputSelectionMethod.MAX_P)
class MaxP(InputSelection):
    """
    Max P input selection method.
    """

    def __init__(self):
        super().__init__(method=InputSelectionMethod.MAX_P)

    def select_input(self):
        raise NotImplementedError("Not implemented yet.")
