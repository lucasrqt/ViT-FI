import enum
import configs
import os
import torch


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


class DSAUtils:
    """
    DSA utils class.
    """

    @staticmethod
    def check_folder(folder):
        """
        Check if the folder exists.
        """

        if not os.path.exists(folder):
            os.makedirs(folder)

    @staticmethod
    def is_ats_present(folder, filename):
        """
        Check if the file is present.
        """
        file = os.path.join(folder, filename)
        return os.path.exists(file)

    @staticmethod
    def construct_file_name(
        model: str, precision: str, layer: str, class_idx: int
    ) -> str:
        """
        Construct the file name.
        """
        return f"{model}-{precision}-layer_{layer}-class_{class_idx}.pt"

    @staticmethod
    def get_hookable_layers(model, layer: str = configs.MLP) -> list:
        """
        Get the hookable layers of the model.
        """
        hookable_layers = []
        for name, module in model.named_modules():
            if module.__class__.__name__.strip() == layer:
                hook = DSAHook(model, layer)
                handler = module.register_forward_hook(hook)
                hookable_layers.append((hook, handler))

        return hookable_layers

    @staticmethod
    def select_layer(layer_handlers: list, layer_idx: int) -> None:
        """
        Select the layer to be used for DSA.
        """
        if layer_idx < 0 or layer_idx >= len(layer_handlers):
            raise ValueError("Layer index out of range.")

        hook, handler = layer_handlers[layer_idx]

        for i, (hook, handler) in enumerate(layer_handlers):
            if i != layer_idx:
                handler.remove()
                del hook

        return hook, handler

    @staticmethod
    def load_ats(path, filename: str) -> torch.Tensor:
        """
        Load the activations traces from the file.
        """
        ats_path = os.path.join(path, filename)
        if os.path.exists(ats_path):
            ats = torch.load(ats_path)
            return ats
        else:
            raise FileNotFoundError(f"File {filename} not found.")

    @staticmethod
    def save_ats(ats: torch.Tensor, folder_path, filename: str) -> None:
        """
        Save the activation traces to the file.
        """
        path = os.path.join(folder_path, filename)
        torch.save(ats, path)


class DSAHook:
    """
    DSA hook class.
    """

    def __init__(self, model, layer):
        self.model: torch.nn.Module = model
        self.ats: torch.Tensor = None
        self.layer_name: str = layer

    def __call__(self, module, input, output):
        """
        Call the hook.
        """
        self.ats = output

    def get_ats(self) -> torch.Tensor:
        """
        Get the activation traces.
        """
        return self.ats

    def clear_ats(self):
        """
        Clear the activation traces.
        """
        self.ats = None
