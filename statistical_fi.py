import torch
import configs
import random
import os
import pandas as pd
from compare_utils import get_top_k_labels
import time
import sys
import numpy as np
import enum

_LAYER_TO_HOOK = [1e-30]
_HOOKABLE_LAYERS = []

_ALTERED_INDICES = None
_REL_ERR_INDICES = None
_RELATIVE_ERRORS = None
_NEG_INF_IDX = None
_POS_INF_IDX = None
_NAN_IDX = None

MODULE, MICROOP_SIZE, INPUT_SIZE = 0, 1, 2

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

class MicroopHook():
    def __init__(self, model_name, microop, batch_size, layer_id, fault_model):
        self.model_name = model_name
        self.microop = microop
        self.layer_id = layer_id
        self.fault_model = fault_model
        self.critical_batches = None
        self.batch_size = batch_size
        self.batch_counter = 0
        self.save_critical_logits = False

    def __process_fault_model(self):
        fault_model = self.fault_model
        altered_floats = fault_model["#alt_val"]
        float_to_nan = fault_model["#nan"]
        nb_neginf = fault_model["#neg_inf"]
        nb_posinf = fault_model["#pos_inf"]

        return (fault_model, altered_floats.item(), float_to_nan.item(), nb_neginf.item(), nb_posinf.item())

    def set_critical_batches(self, critical_batches):
        self.critical_batches = critical_batches

    def set_save_critical_logits(self, save_critical_logits):
        self.save_critical_logits = save_critical_logits


    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        # Move the output to CPU for computations
        faulty_output = module_output.clone().cpu()

        fault_model, altered_floats, float_to_nan, nb_neginf, nb_posinf = self.__process_fault_model()
        nb_total_faults = altered_floats + float_to_nan + nb_neginf + nb_posinf

        altered_floats_ratio = altered_floats / fault_model["#total"].item()
        float_to_nan_ratio = float_to_nan / fault_model["#total"].item()
        nb_neginf_ratio = nb_neginf / fault_model["#total"].item()
        nb_posinf_ratio = nb_posinf / fault_model["#total"].item()
        total_ratio = nb_total_faults / fault_model["#total"].item()
        num_elements = int(total_ratio * faulty_output.numel())

        # Select random elements to modify
        if not hasattr(self, '_altered_indices') or self._altered_indices is None:
            self._altered_indices = torch.randperm(faulty_output.numel())[:num_elements]

        start, end = 0, int(altered_floats_ratio * num_elements)
        rel_err_indices = self._altered_indices[start:end]

        # Get random relative errors
        if not hasattr(self, '_relative_errors') or self._relative_errors is None:
            nb_bins = fault_model.columns.str.startswith('bin_').sum()
            if not hasattr(self, '_bins') or self._bins is None:
                self._bins = torch.tensor(
                    [fault_model[f"bin_{i}"].item() for i in range(nb_bins)]
                )
                counts = torch.tensor(
                    [fault_model[f"hist_{i}"].item() for i in range(nb_bins)], dtype=torch.float
                )
                probs = counts / counts.sum()
                self._probs = probs

        num_rel_errors = int(altered_floats_ratio * num_elements)
        rel_err = self._bins[torch.multinomial(self._probs, num_rel_errors, replacement=True)]
        # rel_err = self._relative_errors[torch.randperm(self._relative_errors.numel())[:int(altered_floats_ratio * num_elements)]]

        faulty_output = faulty_output.flatten()
        faulty_output[rel_err_indices] *= (1 + rel_err)
        faulty_output = faulty_output.view(module_output.shape)

        # Move the output back to the original device
        faulty_output = faulty_output.to(module_output.device)

        self.batch_counter += 1

        return faulty_output

class GetLayerSize():
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



def get_fault_model(fault_model_file, model_name, microop, precision, threshold):
    fault_model_file = os.path.join(configs.RESULTS_DIR, fault_model_file)
    fault_model = pd.read_csv(fault_model_file, index_col=False)
    fault_model = fault_model[
        (fault_model["model"] == model_name)
        & (fault_model["microop"] == microop)
        & (fault_model["precision"] == precision)
        & (fault_model["diff_threshold"] == float(threshold))
    ]

    return fault_model


def check_microop(model_name, microop):
    if model_name in configs.SWIN_MODELS:
        return microop in configs.SWIN_MODULES
    elif model_name in configs.CLASSICAL_VIT_MODELS:
        return microop in configs.VIT_MODULES
    else:
        return ValueError(f"Model {model_name} not supported.")


def select_layer(target: LayerChoice):
    if target == LayerChoice.FIRST:
        return _HOOKABLE_LAYERS[0][MODULE]
    elif target == LayerChoice.MIDDLE:
        return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2][MODULE]
    elif target == LayerChoice.LAST:
        return _HOOKABLE_LAYERS[-1][MODULE]
    else:
        return ValueError("Invalid layer choice.")

def hook_microop(model, model_name, microop, batch_size, fault_model, dummy_input, target) -> torch.utils.hooks.RemovableHandle:
    layers = list()
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
    hook = MicroopHook(model_name, microop, batch_size, layer_id, fault_model)
    handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)

    return hook, handler


def run_inference(model, images, device):
    with torch.no_grad():
        output = model(images)
        if "cuda" in device:
            torch.cuda.synchronize()
        out_top_k = get_top_k_labels(output, configs.TOP_1)
        return out_top_k


