import torch
import configs
import random
import os
import pandas as pd
from compare_utils import get_top_k_labels
import time
import sys
import numpy as np

_LAYER_TO_HOOK = [1e-30]

class MicroopHook():
    def __init__(self, microop, layer_id, fault_model):
        self.microop = microop
        self.layer_id = layer_id
        self.fault_model = fault_model

    def __process_fault_model(self):
        fault_model = self.fault_model
        altered_floats = fault_model["#alt_val"] / fault_model["#total"]
        float_to_nan = fault_model["#nan"] / fault_model["#total"]
        nb_neginf = fault_model["#neg_inf"] / fault_model["#total"]
        nb_posinf = fault_model["#pos_inf"] / fault_model["#total"]
        max_diff = fault_model["Q0%"]
        min_diff = fault_model["Q100%"]

        return (fault_model, altered_floats.item(), float_to_nan.item(), nb_neginf.item(), nb_posinf.item(), max_diff.item(), min_diff.item())
    
    def __sample_relative_errors(self, size, quantiles, percentile_values):
        return np.interp(np.random.rand(size), quantiles, percentile_values)

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        print(f"\n [+] INSIDE HOOK: {module_output.shape}")
        faulty_input = module_output.clone()
        device = module_output.get_device()
        if device == -1:
            device = "cpu"
        else:
            device = f"cuda:{device}"
        
        faulty_input = faulty_input.to(device)

        fault_model, altered_floats_percentage, float_to_nan, nb_neginf, nb_posinf, max_diff, min_diff = self.__process_fault_model()

        # print(f" [+] microop: {self.microop}")
        # print(f" [+] infs = {(nb_posinf + nb_neginf)*100:.6f}% nans = {(float_to_nan * 100):.6}%")
        # print(f" [+] lagre values = {(nb_val_gt_1e30 + nb_val_lt_1e30)*100:.6f}%")
        # sys.exit(0)


        # Randomly select a subset of the coordinates to multiply by the relative error
        # Generate cumulative distribution from percentiles
        percentiles = []
        for i in range(0, 101, 5):
            percentiles.append(fault_model[f"Q{i}%"].item())
        percentile_values = np.array(percentiles[1:-1])  # Exclude Q0% and Q100% for realistic distribution
        quantiles = np.linspace(0, 1, len(percentile_values))

        # Select random elements to modify
        num_elements = int(altered_floats_percentage * faulty_input.numel())
        indices = torch.randperm(faulty_input.numel())[:num_elements].to(device)

        # Get random relative errors
        relative_errors = torch.tensor(self.__sample_relative_errors(num_elements, quantiles, percentile_values), dtype=torch.float32).to(device)

        faulty_input = faulty_input.flatten()
        faulty_input[indices] *= (1 + relative_errors)

        faulty_input = faulty_input.view(module_output.shape).to(device)
        print(f" [+] FAULTY INPUT: {faulty_input.shape}")
        print(f" [+] IS MODIFIED: {not torch.equal(faulty_input, module_output)}")

        # num_modif_neg = int(faulty_input.numel() * nb_neg)
        # num_modif_pos = int(faulty_input.numel() * nb_pos)
        
        # random_neg_indices = torch.randperm(len(negative_coords))[:num_modif_neg].to(device)
        # random_pos_indices = torch.randperm(len(positive_coords))[:num_modif_pos].to(device)
        # flat_tensor = faulty_input.flatten().to(device)

        # flat_tensor[random_neg_indices] = flat_tensor[random_neg_indices] * neg_relative_err
        # flat_tensor[random_pos_indices] = flat_tensor[random_pos_indices] * pos_relative_err

        # faulty_input = flat_tensor.view(faulty_input.shape).to(device)

        # # handle nan and inf
        # zero_coords = torch.nonzero(faulty_input == 0, as_tuple=False).to(device)
        # non_zero_coords = torch.nonzero(faulty_input != 0, as_tuple=False).to(device)
        # all_coords = torch.cat((zero_coords, non_zero_coords), dim=0).to(device)

        # # Remove the coordinates from subset_neg_coords and subset_pos_coords from all_coords
        # mask = ~torch.isin(all_coords, random_neg_indices).all(dim=1).to(device)
        # all_coords = all_coords[mask]
        # mask = ~torch.isin(all_coords, random_pos_indices).all(dim=1).to(device)
        # all_coords = all_coords[mask]

        # random_nan_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * float_to_nan)]
        # subset_nan_coords = all_coords[random_nan_indices]
        # for coord in subset_nan_coords:
        #     coord = tuple(coord)
        #     faulty_input[coord] = float("nan")

        # mask = ~torch.isin(all_coords, subset_nan_coords).all(dim=1).to(device)
        # all_coords = all_coords[mask]

        # random_neginf_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_neginf)].to(device)
        # subset_neginf_coords = all_coords[random_neginf_indices]

        # random_posinf_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_posinf)].to(device)
        # subset_posinf_coords = all_coords[random_posinf_indices]

        # for coord in subset_neginf_coords:
        #     coord = tuple(coord)
        #     faulty_input[coord] = float("-inf")

        # for coord in subset_posinf_coords:
        #     coord = tuple(coord)
        #     faulty_input[coord] = float("inf")

        # mask = ~torch.isin(all_coords, subset_neginf_coords).all(dim=1).to(device)
        # all_coords = all_coords[mask]
        # mask = ~torch.isin(all_coords, subset_posinf_coords).all(dim=1).to(device)
        # all_coords = all_coords[mask]

        # # handle large values
        # random_val_gt_1e30_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_val_gt_1e30)].to(device)
        # subset_val_gt_1e30_coords = all_coords[random_val_gt_1e30_indices]

        # random_val_lt_1e30_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_val_lt_1e30)].to(device)
        # subset_val_lt_1e30_coords = all_coords[random_val_lt_1e30_indices]

        # for coord in subset_val_gt_1e30_coords:
        #     coord = tuple(coord)
        #     low, high = 1e30, max_diff
        #     faulty_input[coord] = torch.rand(1) * (high - low) + low

        # for coord in subset_val_lt_1e30_coords:
        #     coord = tuple(coord)
        #     low, high = min_diff, -1e30
        #     faulty_input[coord] = torch.rand(1) * (high - low) + low

        return faulty_input

class GetLayerSize():
    def __init__(self):
        self.input_size = 0
        self.microop_size = 0

    def hook_fn_to_get_layer_size(self, module, module_input, module_output) -> None:
        global _LAYER_TO_HOOK
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        self.input_size = sum(p.numel() for p in module_input)
        self.microop_size = layer_num_parameters * self.input_size
        if self.microop_size > _LAYER_TO_HOOK[-1]:
            _LAYER_TO_HOOK = [module, self.microop_size, self.input_size]


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


def hook_microop(model, microop, fault_model, dummy_input) -> torch.utils.hooks.RemovableHandle:
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

    layer = _LAYER_TO_HOOK[0]
    hook = MicroopHook(microop, layer_id, fault_model)
    handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)

    return hook, handler


def run_inference(model, images, device):
    with torch.no_grad():
        output = model(images)
        if "cuda" in device:
            torch.cuda.synchronize()
        out_top_k = get_top_k_labels(output, configs.TOP_1)
        return out_top_k


