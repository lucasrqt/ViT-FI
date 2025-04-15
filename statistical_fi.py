import torch
import configs
import os
import pandas as pd
from utils.compare_utils import get_top_k_labels, get_top_k_probs
import enum

_LAYER_TO_HOOK = [1e-30]
_HOOKABLE_LAYERS = []

MODULE, MICROOP_SIZE, INPUT_SIZE = 0, 1, 2


class InjectionType(enum.Enum):
    RANDOM = 0
    FIXED = 1

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class LayerChoice(enum.Enum):
    FIRST = 0
    MIDDLE = 1
    LAST = 2
    SMALLEST = 3
    LARGEST = 4

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class MicroopHook:
    """
    Class to inject faults into a specific micro-operation in a neural network model.

    Args:
        model_name (str): Name of the model.
        microop (str): Name of the micro-operation to inject faults into.
        batch_size (int): Size of the batch.
        layer_id (int): ID of the layer to inject faults into.
        fault_model (pd.DataFrame): Fault model containing information about faults.

    Attributes:
        model_name (str): Name of the model.
        microop (str): Name of the micro-operation to inject faults into.
        layer_id (int): ID of the layer to inject faults into.
        fault_model (pd.DataFrame): Fault model containing information about faults.
        critical_batches (list): List of critical batches to inject faults into.
        batch_size (int): Size of the batch.
        batch_counter (int): Counter for the number of batches processed.
        save_critical_logits (bool): Flag to save critical logits.

    Methods:
        __process_fault_model(): Processes the fault model to extract relevant information.
        set_critical_batches(critical_batches): Sets the critical batches for fault injection.
        set_save_critical_logits(save_critical_logits): Sets the flag to save critical logits.
        hook_fn_to_inject_fault(module, module_input, module_output): Hook function to inject faults into the model.
    """

    def __init__(
        self,
        model_name,
        microop,
        batch_size,
        nb_inputs,
        layer_id,
        fault_model,
        inj_type,
    ):
        """
        Initializes the MicroopHook class.
        """
        self.model_name = model_name
        self.microop = microop
        self.layer_id = layer_id
        self.fault_model = fault_model
        self.batch_size = batch_size
        self.last_batch_size = nb_inputs % self.batch_size
        self.save_critical_logits = False
        self.injection_type = inj_type

    def __process_fault_model(self) -> tuple:
        """
        Processes the fault model to extract relevant information.
        Returns:
            tuple: A tuple containing the fault model and various fault counts.
        """
        fault_model = self.fault_model
        altered_floats = fault_model["#alt_val"]
        float_to_nan = fault_model["#nan"]
        nb_neginf = fault_model["#neg_inf"]
        nb_posinf = fault_model["#pos_inf"]

        return (
            fault_model,
            altered_floats.item(),
            float_to_nan.item(),
            nb_neginf.item(),
            nb_posinf.item(),
        )

    def set_critical_batches(self, critical_batches) -> None:
        self.critical_batches = critical_batches

    def set_save_critical_logits(self, save_critical_logits) -> None:
        self.save_critical_logits = save_critical_logits

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        """
        Hook function to inject faults into the model.
        Args:
            module: The module to inject faults into.
            module_input: The input to the module.
            module_output: The output of the module.
        """

        # Move the output to CPU for computations
        faulty_output = module_output.clone().cpu()

        # gathering the fault model
        fault_model, altered_floats, float_to_nan, nb_neginf, nb_posinf = (
            self.__process_fault_model()
        )
        nb_total_faults = altered_floats + float_to_nan + nb_neginf + nb_posinf

        altered_floats_ratio = altered_floats / fault_model["#total"].item()
        # float_to_nan_ratio = float_to_nan / fault_model["#total"].item()
        # nb_neginf_ratio = nb_neginf / fault_model["#total"].item()
        # nb_posinf_ratio = nb_posinf / fault_model["#total"].item()
        total_ratio = nb_total_faults / fault_model["#total"].item()
        num_elements = int(total_ratio * faulty_output.numel())

        ### V1 with random error sampling and fixed positions
        # # Select random elements to modify and generate it only once
        # if not hasattr(self, "_altered_indices") or self._altered_indices is None:
        #     self._altered_indices = torch.randperm(faulty_output.numel())[:num_elements]

        # num_rel_errors = int(altered_floats_ratio * num_elements)
        # rel_err_indices = self._altered_indices[:num_rel_errors]

        # # Get random relative errors
        # if not hasattr(self, "_relative_errors") or self._relative_errors is None:
        #     nb_bins = fault_model.columns.str.startswith("bin_").sum()
        #     if not hasattr(self, "_bins") or self._bins is None:
        #         self._bins = torch.tensor(
        #             [fault_model[f"bin_{i}"].item() for i in range(nb_bins)]
        #         )
        #         counts = torch.tensor(
        #             [fault_model[f"hist_{i}"].item() for i in range(nb_bins)],
        #             dtype=torch.float,
        #         )
        #         probs = counts / counts.sum()
        #         self._probs = probs

        # rel_err = self._bins[
        #     torch.multinomial(self._probs, num_rel_errors, replacement=True)
        # ]

        ### V2 with fixed error sampling and random positions
        # # Select random faults to inject
        # num_rel_errors = int(altered_floats_ratio * num_elements)
        # if not hasattr(self, "_relative_errors") or self._relative_errors is None:
        #     nb_bins = fault_model.columns.str.startswith("bin_").sum()
        #     if not hasattr(self, "_bins") or self._bins is None:
        #         self._bins = torch.tensor(
        #             [fault_model[f"bin_{i}"].item() for i in range(nb_bins)]
        #         )
        #         counts = torch.tensor(
        #             [fault_model[f"hist_{i}"].item() for i in range(nb_bins)],
        #             dtype=torch.float,
        #         )
        #         probs = counts / counts.sum()
        #         self._probs = probs

        #     self._relative_errors = self._bins[
        #         torch.multinomial(self._probs, num_rel_errors, replacement=True)
        #     ]

        # rel_err = self._relative_errors

        # rel_err_indices = torch.randperm(faulty_output.numel())[:num_rel_errors]

        ### V3 with random error sampling and random positions
        if self.injection_type == InjectionType.RANDOM:
            num_rel_errors = int(altered_floats_ratio * num_elements)

            # Get random relative errors
            if not hasattr(self, "_relative_errors") or self._relative_errors is None:
                nb_bins = fault_model.columns.str.startswith("bin_").sum()
                if not hasattr(self, "_bins") or self._bins is None:
                    self._bins = torch.tensor(
                        [fault_model[f"bin_{i}"].item() for i in range(nb_bins)]
                    )
                    counts = torch.tensor(
                        [fault_model[f"hist_{i}"].item() for i in range(nb_bins)],
                        dtype=torch.float,
                    )
                    probs = counts / counts.sum()
                    self._probs = probs

            rel_err = self._bins[
                torch.multinomial(self._probs, num_rel_errors, replacement=True)
            ]
            rel_err_indices = torch.randperm(faulty_output.numel())[:num_rel_errors]

        elif self.injection_type == InjectionType.FIXED:
            num_rel_errors = int(altered_floats_ratio * num_elements)

            if not hasattr(self, "_altered_indices") or self._altered_indices is None:
                self._altered_indices = torch.randperm(faulty_output.numel())[
                    :num_rel_errors
                ]
                self._last_batch_msk = self._altered_indices < (
                    faulty_output.numel() // self.batch_size * self.last_batch_size
                )
                self._split1_indices = self._altered_indices[self._last_batch_msk]

            rel_err_indices = self._altered_indices

            if not hasattr(self, "_relative_errors") or self._relative_errors is None:
                nb_bins = fault_model.columns.str.startswith("bin_").sum()
                if not hasattr(self, "_bins") or self._bins is None:
                    self._bins = torch.tensor(
                        [fault_model[f"bin_{i}"].item() for i in range(nb_bins)]
                    )
                    counts = torch.tensor(
                        [fault_model[f"hist_{i}"].item() for i in range(nb_bins)],
                        dtype=torch.float,
                    )
                    probs = counts / counts.sum()
                    self._probs = probs

                self._relative_errors = self._bins[
                    torch.multinomial(self._probs, num_rel_errors, replacement=True)
                ]

            rel_err = self._relative_errors

            if faulty_output.shape[0] == self.last_batch_size:
                rel_err_indices = self._split1_indices
                rel_err = self._relative_errors[self._last_batch_msk]

        faulty_output = faulty_output.flatten()
        faulty_output[rel_err_indices] *= 1 + rel_err
        faulty_output = faulty_output.view(module_output.shape)

        # Move the output back to the original device
        faulty_output = faulty_output.to(module_output.device)

        return faulty_output


class GetLayerSize:
    """Class to get the size of a layer in a neural network model.
    This class is used to calculate the size of a layer in terms of the number of parameters and input size.
    It is used in the context of fault injection in neural networks.
    """

    def __init__(self):
        self.input_size = 0
        self.microop_size = 0

    def hook_fn_to_get_layer_size(self, module, module_input, module_output) -> None:
        # global _LAYER_TO_HOOK
        global _HOOKABLE_LAYERS
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        self.input_size = sum(p.numel() for p in module_input)
        self.microop_size = layer_num_parameters * self.input_size

        _HOOKABLE_LAYERS.append((module, self.microop_size, self.input_size))

        # if self.microop_size > _LAYER_TO_HOOK[-1]:
        # _LAYER_TO_HOOK = [module, self.microop_size, self.input_size]


def get_fault_model(
    fault_model_file, model_name, microop, precision, threshold
) -> pd.DataFrame:
    """
    Get the fault model for a specific model, microop, precision, and threshold.
    Args:
        fault_model_file (str): Path to the fault model file.
        model_name (str): Name of the model.
        microop (str): Name of the micro-operation.
        precision (str): Precision of the model.
        threshold (float): Threshold for filtering the fault model.
    Returns:
        pd.DataFrame: Filtered fault model DataFrame.
    """
    fault_model_file = os.path.join(configs.RESULTS_DIR, fault_model_file)
    fault_model = pd.read_csv(fault_model_file, index_col=False)
    fault_model = fault_model[
        (fault_model["model"] == model_name)
        & (fault_model["microop"] == microop)
        & (fault_model["precision"] == precision)
        & (fault_model["diff_threshold"] == float(threshold))
    ]

    return fault_model


def check_microop(model_name, microop) -> bool:
    """
    Check if the micro-operation is valid for the given model.
    Args:
        model_name (str): Name of the model.
        microop (str): Name of the micro-operation.
    Returns:
        bool: True if the micro-operation is valid, otherwise raises a ValueError.
    """
    if model_name in configs.SWIN_MODELS:
        return microop in configs.SWIN_MODULES
    elif model_name in configs.CLASSICAL_VIT_MODELS:
        return microop in configs.VIT_MODULES
    else:
        return ValueError(f"Model {model_name} not supported.")


def select_layer(target: LayerChoice) -> torch.nn.Module:
    """
    Selects a layer based on the target choice.
    Args:
        target (LayerChoice): The target choice for selecting the layer.
    Returns:
        torch.nn.Module: The selected layer.
    """
    if target == LayerChoice.FIRST:
        return _HOOKABLE_LAYERS[0][MODULE]
    elif target == LayerChoice.MIDDLE:
        return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2][MODULE]
    elif target == LayerChoice.LAST:
        return _HOOKABLE_LAYERS[-1][MODULE]
    else:
        return ValueError("Invalid layer choice.")


def hook_microop(
    model,
    model_name,
    microop,
    batch_size,
    nb_inputs,
    fault_model,
    dummy_input,
    target,
    injection_type,
) -> torch.utils.hooks.RemovableHandle:
    """
    Hook a specific micro-operation in the model to inject faults.
    Args:
        model (torch.nn.Module): The model to hook.
        model_name (str): Name of the model.
        microop (str): Name of the micro-operation to hook.
        batch_size (int): Size of the batch.
        fault_model (pd.DataFrame): Fault model DataFrame.
        dummy_input (torch.Tensor): Dummy input tensor for the model.
        target (LayerChoice): Target choice for selecting the layer.
    Returns:
        tuple: A tuple containing the hook and handler.
    """
    # layers = list()
    handlers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            # layers.append((layer, layer_id))
            hook = GetLayerSize()
            handler = layer.register_forward_hook(hook.hook_fn_to_get_layer_size)
            handlers.append(handler)

    _ = model(dummy_input)

    for handler in handlers:
        handler.remove()

    layer = select_layer(target)
    hook = MicroopHook(
        model_name,
        microop,
        batch_size,
        nb_inputs,
        layer_id,
        fault_model,
        injection_type,
    )
    handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)

    return hook, handler


def run_inference(model, images, device):
    with torch.no_grad():
        output = model(images)
        if "cuda" in device:
            torch.cuda.synchronize()
        out_top_k = get_top_k_labels(output, configs.TOP_1)
        out_top_k_prob = get_top_k_probs(output, configs.TOP_2)
        return out_top_k, out_top_k_prob
