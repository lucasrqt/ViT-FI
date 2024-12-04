import torch
import configs
import random
import os
import pandas as pd
from compare_utils import get_top_k_labels

class MicroopHook():
    def __init__(self, microop, layer_id, fault_model):
        self.microop = microop
        self.layer_id = layer_id
        self.fault_model = fault_model

    def __process_fault_model(self):
        fault_model = self.fault_model
        altered_floats = fault_model["# float"] / fault_model["# total"]
        float_to_nan = fault_model["# nan"] / fault_model["# total"]
        nb_neginf = fault_model["# -inf"] / fault_model["# total"]
        nb_posinf = fault_model["# +inf"] / fault_model["# total"]
        nb_val_lt_0 = fault_model["# val < 0 and > -1e3"] / fault_model["# total"]
        nb_val_lt_1e3 = fault_model["# val <= -1e3 and > -1M"] / fault_model["# total"]
        nb_val_lt_1M = fault_model["# val <= -1M and > -1B"] / fault_model["# total"]
        nb_val_lt_1B = fault_model["# val <= -1B and > -1e20"] / fault_model["# total"]
        nb_val_lt_1e20 = fault_model["# val <= -1e20 and > -1e30"] / fault_model["# total"]
        nb_val_lt_1e30 = fault_model["# val <= -1e30"] / fault_model["# total"]
        nb_val_gt_0 = fault_model["# val >= 0 and < 1e3"] / fault_model["# total"]
        nb_val_gt_1e3 = fault_model["# val >= 1e3 and < 1M"] / fault_model["# total"]
        nb_val_gt_1M = fault_model["# val >= 1M and < 1B"] / fault_model["# total"]
        nb_val_gt_1B = fault_model["# val >= 1B and < 1e20"] / fault_model["# total"]
        nb_val_gt_1e20 = fault_model["# val >= 1e20 and < 1e30"] / fault_model["# total"]
        nb_val_gt_1e30 = fault_model["# val > 1e30"] / fault_model["# total"]
        pos_relative_err = fault_model["mean RE > 0"] + 1
        neg_relative_err = abs(fault_model["mean RE < 0"] - 1)
        nb_neg = fault_model["len(RE < 0)"]
        nb_pos = fault_model["len(RE > 0)"]
        max_diff = fault_model["max_diff"]
        min_diff = fault_model["min_diff"]

        return (altered_floats, float_to_nan, nb_neginf, nb_posinf, nb_val_lt_0, nb_val_lt_1e3, nb_val_lt_1M, nb_val_lt_1B, nb_val_lt_1e20, nb_val_lt_1e30, nb_val_gt_0, nb_val_gt_1e3, nb_val_gt_1M, nb_val_gt_1B, nb_val_gt_1e20, nb_val_gt_1e30, pos_relative_err, neg_relative_err, nb_neg, nb_pos, max_diff, min_diff)

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        faulty_input = module_input[0].clone()
        altered_floats, float_to_nan, nb_neginf, nb_posinf, nb_val_lt_0, nb_val_lt_1e3, nb_val_lt_1M, nb_val_lt_1B, nb_val_lt_1e20, nb_val_lt_1e30, nb_val_gt_0, nb_val_gt_1e3, nb_val_gt_1M, nb_val_gt_1B, nb_val_gt_1e20, nb_val_gt_1e30, pos_relative_err, neg_relative_err, nb_neg, nb_pos, max_diff, min_diff = self.__process_fault_model()
        altered_floats = altered_floats.item()
        float_to_nan = float_to_nan.item()
        nb_neginf = nb_neginf.item()
        nb_posinf = nb_posinf.item()
        nb_val_lt_0 = nb_val_lt_0.item()
        nb_val_lt_1e3 = nb_val_lt_1e3.item()
        nb_val_lt_1M = nb_val_lt_1M.item()
        nb_val_lt_1B = nb_val_lt_1B.item()
        nb_val_lt_1e20 = nb_val_lt_1e20.item()
        nb_val_lt_1e30 = nb_val_lt_1e30.item()
        nb_val_gt_0 = nb_val_gt_0.item()
        nb_val_gt_1e3 = nb_val_gt_1e3.item()
        nb_val_gt_1M = nb_val_gt_1M.item()
        nb_val_gt_1B = nb_val_gt_1B.item()
        nb_val_gt_1e20 = nb_val_gt_1e20.item()
        nb_val_gt_1e30 = nb_val_gt_1e30.item()
        pos_relative_err = pos_relative_err.item()
        neg_relative_err = neg_relative_err.item()
        nb_neg = nb_neg.item()
        nb_pos = nb_pos.item()
        max_diff = max_diff.item()
        min_diff = min_diff.item()

        negative_coords = torch.nonzero(faulty_input < 0, as_tuple=False)
        positive_coords = torch.nonzero(faulty_input > 0, as_tuple=False)

        # Randomly select a subset of the coordinates to multiply by the relative error
        random_neg_indices = torch.randperm(len(negative_coords))[:int(faulty_input.numel() * nb_neg)]
        random_pos_indices = torch.randperm(len(positive_coords))[:int(faulty_input.numel() * nb_pos)]

        subset_neg_coords = negative_coords[random_neg_indices]
        subset_pos_coords = positive_coords[random_pos_indices]

        for coord in subset_neg_coords:
            coord = tuple(coord)
            faulty_input[coord] = faulty_input[coord] * neg_relative_err

        for coord in subset_pos_coords:
            coord = tuple(coord)
            faulty_input[coord] = faulty_input[coord] * pos_relative_err

        # handle nan and inf
        zero_coords = torch.nonzero(faulty_input == 0, as_tuple=False)
        non_zero_coords = torch.nonzero(faulty_input != 0, as_tuple=False)
        all_coords = torch.cat((zero_coords, non_zero_coords), dim=0)

        # Remove the coordinates from subset_neg_coords and subset_pos_coords from all_coords
        mask = ~torch.isin(all_coords, subset_neg_coords).all(dim=1)
        all_coords = all_coords[mask]
        mask = ~torch.isin(all_coords, subset_pos_coords).all(dim=1)
        all_coords = all_coords[mask]

        random_nan_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * float_to_nan)]
        subset_nan_coords = all_coords[random_nan_indices]
        for coord in subset_nan_coords:
            coord = tuple(coord)
            faulty_input[coord] = float("nan")

        mask = ~torch.isin(all_coords, subset_nan_coords).all(dim=1)
        all_coords = all_coords[mask]

        random_neginf_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_neginf)]
        subset_neginf_coords = all_coords[random_neginf_indices]

        random_posinf_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_posinf)]
        subset_posinf_coords = all_coords[random_posinf_indices]

        for coord in subset_neginf_coords:
            coord = tuple(coord)
            faulty_input[coord] = float("-inf")

        for coord in subset_posinf_coords:
            coord = tuple(coord)
            faulty_input[coord] = float("inf")

        mask = ~torch.isin(all_coords, subset_neginf_coords).all(dim=1)
        all_coords = all_coords[mask]
        mask = ~torch.isin(all_coords, subset_posinf_coords).all(dim=1)
        all_coords = all_coords[mask]

        # handle large values
        random_val_gt_1e30_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_val_gt_1e30)]
        subset_val_gt_1e30_coords = all_coords[random_val_gt_1e30_indices]

        random_val_lt_1e30_indices = torch.randperm(len(all_coords))[:int(faulty_input.numel() * nb_val_lt_1e30)]
        subset_val_lt_1e30_coords = all_coords[random_val_lt_1e30_indices]

        for coord in subset_val_gt_1e30_coords:
            coord = tuple(coord)
            low, high = 1e30, max_diff
            faulty_input[coord] = torch.rand(1) * (high - low) + low

        for coord in subset_val_lt_1e30_coords:
            coord = tuple(coord)
            low, high = min_diff, -1e30
            faulty_input[coord] = torch.rand(1) * (high - low) + low

        module_input = (faulty_input,)


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


def hook_microop(model, microop, fault_model) -> torch.utils.hooks.RemovableHandle:
    layers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            layers.append((layer, layer_id))
            print(f" [+] Layer {layer_id}")

    random.seed(configs.SEED)
    layer_index = random.randint(0, len(layers) - 1)

    layer, layer_id = layers[layer_index]
    hook = MicroopHook(microop, layer_id, fault_model)
    handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)

    return handler


def run_inference(model, images, device):
    with torch.no_grad():
        output = model(images)
        if "cuda" in device:
            torch.cuda.synchronize()
        out_top_k = get_top_k_labels(output, configs.TOP_1)
        return out_top_k


