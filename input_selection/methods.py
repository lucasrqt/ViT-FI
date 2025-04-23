from utils.input_selection import InputSelectionMethod


class InputSelection:
    def __init__(self, method: InputSelectionMethod):
        self.method = method

    def select_input(self):
        raise NotImplementedError(
            "Abstract class method called. Call a subclass method instead."
        )


class InputSelectionFactory:
    """
    Factory class for creating input selection methods.
    """

    _registery = {}

    @classmethod
    def register(cls, method: InputSelectionMethod):
        """
        Register a new input selection method.
        """

        def decorator(subclass):
            cls._registery[method] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, method: InputSelectionMethod, *args, **kwargs):
        """
        Create an input selection method.
        """
        if method not in cls._registery:
            raise ValueError(f"Input selection method {method} not registered")
        return cls._registery[method](*args, **kwargs)
