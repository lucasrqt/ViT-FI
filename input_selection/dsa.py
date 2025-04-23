from methods import InputSelection, InputSelectionFactory
from utils.input_selection import InputSelectionMethod


@InputSelectionFactory.register(InputSelectionMethod.DSA)
class DSA(InputSelection):
    """
    DSA input selection method.

    Adapted from the code of the paper "Guiding Deep Learning System Testing using Surprise Adequacy" (https://doi.org/10.1109/ICSE.2019.00108)
    source code available at: https://github.com/coinse/sadl/blob/master/sa.py
    """

    def __init__(self):
        super().__init__(method=InputSelectionMethod.DSA)

    def __get_ats(self):
        """
        Get the attention scores of the model.
        """
        raise NotImplementedError("Not implemented yet.")

    def __get_trained_target_ats(self):
        """
        Get the trained target attention scores of the model.
        """
        raise NotImplementedError("Not implemented yet.")

    def __fetch_dsa(self):
        """
        Fetch the DSA scores of the model.
        """
        raise NotImplementedError("Not implemented yet.")

    def find_closest_at(self):
        """
        Find the closest attention scores to the trained target attention scores.
        """
        raise NotImplementedError("Not implemented yet.")

    def select_input(self):
        raise NotImplementedError("Not implemented yet.")
