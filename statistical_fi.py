import torch
import configs
import os
import pandas as pd
from utils.compare_utils import get_top_k_labels, get_top_k_probs
import enum
import numpy as np
import struct

_LAYER_TO_HOOK = [1e-30]
_HOOKABLE_LAYERS = []

MODULE, MICROOP_SIZE, INPUT_SIZE, LAYER_FULL_NAME = 0, 1, 2, 3

DEFAULT_CORRUPT_VALUE = 1e-4
DEFAULT_COL, DEFAULT_ROW = 0, 0
DEFAULT_BIT_POSITION = 23  # Flipping the last bit of the exponent in float32

class InjectionType(enum.Enum):
    RANDOM = 0
    FIXED = 1
    SINGLE = 2
    ROW = 3
    COL = 4
    BULLET_WAKE = 5 # ONLY for 3D microop like one of Swin
    SINGLE_RANDOM = 6
    MULTIPLE_RANDOM = 7

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class LayerChoice(enum.Enum):
    FIRST = 0
    FIRST_HALF = 1
    MIDDLE = 2
    MIDDLE_HALF = 3
    BEFORE_LAST = 4
    LAST = 5
    SMALLEST = 6
    LARGEST = 7

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)


class RangeRestrictionMode(enum.Enum):
    NONE = 0
    CLAMP = 1
    TO_ZERO = 2

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
        layer_full_name,
        seed=0,
        bit_position=DEFAULT_BIT_POSITION,
        dataset_name=None,
        rr_mode=RangeRestrictionMode.NONE,
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
        self.layer_full_name = layer_full_name
        self.seed = seed
        self.bit_position = bit_position
        self.dataset_name = dataset_name
        self.rr_min_value = None
        self.rr_max_value = None
        self.rr_mode = rr_mode

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
    
    def get_relative_errors(self):
        return self._relative_errors if hasattr(self, "_relative_errors") else None

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        """
        Hook function to inject faults into the model.
        Args:
            module: The module to inject faults into.
            module_input: The input to the module.
            module_output: The output of the module.
        """
        # Move the output to CPU for computations
        module_output_device = None
        if self.model_name in configs.TEXT_MODELS and isinstance(module_output, tuple):
            module_output_device = module_output[0].device
            faulty_output = module_output[0].clone().cpu()
        else:
            module_output_device = module_output.device
            faulty_output = module_output.clone().cpu()

        module_output_shape = faulty_output.shape

        # Gathering the fault model
        fault_model, altered_floats, float_to_nan, nb_neginf, nb_posinf = (
            self.__process_fault_model()
        )
        nb_total_faults = altered_floats + float_to_nan + nb_neginf + nb_posinf

        altered_floats_ratio = altered_floats / fault_model["#total"].item()
        float_to_nan_ratio = float_to_nan / fault_model["#total"].item()
        nb_neginf_ratio = nb_neginf / fault_model["#total"].item()
        nb_posinf_ratio = nb_posinf / fault_model["#total"].item()
        total_ratio = nb_total_faults / fault_model["#total"].item()
        num_elements = int(total_ratio * faulty_output.numel())

        # min_float, max_float = 1e-8, 1e2
        min_float, max_float = torch.finfo(torch.float32).min, torch.finfo(torch.float32).max

        ### V3 with random error sampling and random positions
        if self.injection_type == InjectionType.RANDOM:
            raise NotImplementedError(
                "Random injection type is not implemented yet. "
                "Please use the FIXED injection type."
            )
            # num_rel_errors = int(altered_floats_ratio * num_elements)
            # num_nan = int(float_to_nan_ratio * num_elements)
            # num_neginf = int(nb_neginf_ratio * num_elements)
            # num_posinf = int(nb_posinf_ratio * num_elements)

            # # --- Relative errors ---
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
            #         self._probs = counts / counts.sum()

            # rel_err = self._bins[
            #     torch.multinomial(self._probs, num_rel_errors, replacement=True)
            # ]
            # rel_err_indices = torch.randperm(faulty_output.numel())[:num_rel_errors]
            # used_indices = set(rel_err_indices.tolist())

            # # Apply relative errors
            # faulty_output[rel_err_indices] *= 1 + rel_err

            # # --- NaNs ---
            # remaining_indices = list(set(range(faulty_output.numel())) - used_indices)
            # nan_indices = torch.tensor(remaining_indices)[
            #     torch.randperm(len(remaining_indices))[:num_nan]
            # ]
            # used_indices.update(nan_indices.tolist())
            # faulty_output[nan_indices] = float('nan')

            # # --- -inf ---
            # remaining_indices = list(set(range(faulty_output.numel())) - used_indices)
            # neginf_indices = torch.tensor(remaining_indices)[
            #     torch.randperm(len(remaining_indices))[:num_neginf]
            # ]
            # used_indices.update(neginf_indices.tolist())
            # faulty_output[neginf_indices] = float('-inf')

            # # --- +inf ---
            # remaining_indices = list(set(range(faulty_output.numel())) - used_indices)
            # posinf_indices = torch.tensor(remaining_indices)[
            #     torch.randperm(len(remaining_indices))[:num_posinf]
            # ]
            # used_indices.update(posinf_indices.tolist())
            # faulty_output[posinf_indices] = float('inf')

        elif self.injection_type == InjectionType.FIXED:
            num_rel_errors = int(altered_floats_ratio * num_elements)
            # num_nan = 341606
            # num_neginf = int(1368 / 2)
            # num_posinf = int(1368 / 2)
            num_nan = 0
            num_neginf = 0
            num_posinf = 0
            # sum_num_err = num_rel_errors + num_nan + num_neginf + num_posinf
            sum_num_err = num_rel_errors + num_neginf + num_posinf

            if not hasattr(self, "_altered_indices") or self._altered_indices is None:
                self._altered_indices = torch.randperm(faulty_output.numel())[:sum_num_err]
                self._last_batch_msk = self._altered_indices < (
                    faulty_output.numel() // self.batch_size * self.last_batch_size
                )
                self._split1_indices = self._altered_indices[self._last_batch_msk]

            rel_err_indices = self._altered_indices[:num_rel_errors]
            nan_indices = self._altered_indices[num_rel_errors:num_rel_errors + num_nan]
            neginf_indices = self._altered_indices[
                num_rel_errors + num_nan:num_rel_errors + num_nan + num_neginf
            ]
            posinf_indices = self._altered_indices[
                num_rel_errors + num_nan + num_neginf:num_rel_errors + num_nan + num_neginf + num_posinf
            ]

            if not hasattr(self, "_relative_errors") or self._relative_errors is None:
                nb_bins = fault_model.columns.str.startswith("bin_").sum()
                if not hasattr(self, "_bins") or self._bins is None:
                    
                    #### if all fault model is considered
                    self._bins = torch.tensor(
                        ([fault_model[f"bin_{i}"].item() for i in range(nb_bins)])
                    )
                    counts = torch.tensor(
                        ([fault_model[f"hist_{i}"].item() for i in range(nb_bins)]),
                        dtype=torch.float,
                    )

                   
                    #### if threshing on relative error selection
                    ## keep only bins and probabilities that have a relative error between -30% and 30%
                    ## Filter out bins with absolute value > 0.3
                    # bin_cols = [col for col in fault_model.columns if col.startswith("bin_")]
                    # for col in bin_cols:
                    #     if abs(fault_model[col].item()) > 0.3:
                    #         # Get the corresponding hist column
                    #         hist_col = col.replace("bin_", "hist_")
                    #         fault_model = fault_model.drop(columns=[col, hist_col])
                    
                    # # Get actual bin column names after filtering
                    # bin_cols = [col for col in fault_model.columns if col.startswith("bin_")]
                    # hist_cols = [col for col in fault_model.columns if col.startswith("hist_")]
                    
                    # self._bins = torch.tensor([fault_model[col].item() for col in bin_cols])
                    # counts = torch.tensor(
                    #     [fault_model[col].item() for col in hist_cols],
                    #     dtype=torch.float,
                    # )
                    self._probs = counts / counts.sum()

                self._relative_errors = self._bins[
                    torch.multinomial(self._probs, sum_num_err, replacement=True)
                ]
                # torch.save(self._relative_errors, f"relerr-{self.model_name}-{self.microop}-layer_{self.layer_id}-seed_{self.seed}.pt")

            err_indices = self._altered_indices

            rel_err = self._relative_errors
            if faulty_output.shape[0] == self.last_batch_size:
                err_indices = self._split1_indices
                rel_err = self._relative_errors[self._last_batch_msk]
                nan_indices = nan_indices[self._last_batch_msk]
                neginf_indices = neginf_indices[self._last_batch_msk]
                posinf_indices = posinf_indices[self._last_batch_msk]

            faulty_output = faulty_output.flatten()

            # finite_mask = torch.isfinite(rel_err)
            # valid_idx = err_indices[finite_mask]
            # valid_rel = rel_err[finite_mask]

            # faulty_output[valid_idx] *= 1 + valid_rel
            # faulty_output[err_indices] = faulty_output[err_indices] + (faulty_output[err_indices] * rel_err)
            faulty_output[err_indices] *= 1 + rel_err

            # faulty_output[nan_indices] = np.nan
            # faulty_output[neginf_indices] = np.NINF
            # faulty_output[posinf_indices] = np.PINF

        # print(f"Injected {num_rel_errors} relative errors, "
        #       f"{num_nan} NaNs,"
        #     #   f"{num_neginf} -inf, and {num_posinf} +inf into the output."
        #       )
        
        # print(f"{ rel_err_indices =  }"
        #       f"{ nan_indices =  }"
        #       f"{ neginf_indices =  }"
        #       f"{ posinf_indices =  }")
        

        # elif self.injection_type == InjectionType.SINGLE:
        #     # corrupt a single value inside tensor
        #     faulty_output = faulty_output.flatten()
        #     corrupt_index = np.random.randint(0, faulty_output.numel())
        #     faulty_output[corrupt_index] *= 1 +DEFAULT_CORRUPT_VALUE
        
        elif self.injection_type == InjectionType.ROW:
            # ROW fault: corrupt same "row" across all batch samples
            # Supports:
            #   3D: [B, H, W]
            #   4D: [B, H, W, C]
            
            original_shape = faulty_output.shape
            dim = len(original_shape)
            
            if dim not in (3, 4):
                raise ValueError(f"ROW injection expects 3D or 4D tensor, got {original_shape}")

            if dim == 3:
                batch_size, row_len, col_len = original_shape

                # Select row to corrupt
                if not hasattr(self, "_selected_row") or self._selected_row is None:
                    self._selected_row = np.random.randint(0, row_len)
                row_idx = self._selected_row

                # Relative errors per column
                if not hasattr(self, "_row_fault_values") or self._row_fault_values is None:
                    self._row_fault_values = torch.empty(col_len, dtype=torch.float, device=faulty_output.device)
                    tmp = torch.empty_like(self._row_fault_values, dtype=torch.float64)
                    tmp.uniform_(float(min_float), float(max_float))
                    self._row_fault_values.copy_(tmp.to(torch.float32))

                faulty_output[:, row_idx, :] = self._row_fault_values

            else:  # 4D case [B, H, W, C]
                batch_size, H, W, C = original_shape

                # Select column (W) and depth (C)
                if not hasattr(self, "_selected_w") or self._selected_w is None:
                    self._selected_w = np.random.randint(0, W)
                if not hasattr(self, "_selected_c") or self._selected_c is None:
                    self._selected_c = np.random.randint(0, C)
                w_idx = self._selected_w
                c_idx = self._selected_c

                # Draw relative errors for each row (H dimension)
                if not hasattr(self, "_row_fault_values_4d") or self._row_fault_values_4d is None:
                    self._row_fault_values_4d = torch.empty(H, dtype=torch.float, device=faulty_output.device)
                    tmp = torch.empty_like(self._row_fault_values_4d, dtype=torch.float64)
                    tmp.uniform_(float(min_float), float(max_float))
                    self._row_fault_values_4d.copy_(tmp.to(torch.float32))

                faulty_output[:, :, w_idx, c_idx] = self._row_fault_values_4d

        # -----------------------------------------------------------

        elif self.injection_type == InjectionType.COL:
            # COL fault: corrupt same "column" across all batch samples
            # Supports:
            #   3D: [B, H, W]
            #   4D: [B, H, W, C]
            
            module_output_shape = faulty_output.shape
            dim = len(module_output_shape)
            
            if dim not in (3, 4):
                raise ValueError(f"COL injection expects 3D or 4D tensor, got {module_output_shape}")

            if dim == 3:
                batch_size, row_len, col_len = module_output_shape

                # Select column to corrupt
                if not hasattr(self, "_selected_col") or self._selected_col is None:
                    self._selected_col = np.random.randint(0, col_len)
                col_idx = self._selected_col

                # Relative errors per row
                if not hasattr(self, "_col_fault_values") or self._col_fault_values is None:
                    self._col_fault_values = torch.empty(row_len, dtype=torch.float, device=faulty_output.device)
                    tmp = torch.empty_like(self._col_fault_values, dtype=torch.float64)
                    tmp.uniform_(float(min_float), float(max_float))
                    self._col_fault_values.copy_(tmp.to(torch.float32))

                faulty_output[:, :, col_idx] = self._col_fault_values

            else:  # 4D case [B, H, W, C]
                batch_size, H, W, C = module_output_shape

                # Select row (H) and depth (C)
                if not hasattr(self, "_selected_h") or self._selected_h is None:
                    self._selected_h = np.random.randint(0, H)
                if not hasattr(self, "_selected_c") or self._selected_c is None:
                    self._selected_c = np.random.randint(0, C)
                h_idx = self._selected_h
                c_idx = self._selected_c

                # Draw relative errors for each column (W dimension)
                if not hasattr(self, "_col_fault_values_4d") or self._col_fault_values_4d is None:
                    self._col_fault_values_4d = torch.empty(W, dtype=torch.float, device=faulty_output.device)
                    tmp = torch.empty_like(self._col_fault_values_4d, dtype=torch.float64)
                    tmp.uniform_(float(min_float), float(max_float))
                    self._col_fault_values_4d.copy_(tmp.to(torch.float32))

                faulty_output[:, h_idx, :, c_idx] = self._col_fault_values_4d


        elif self.injection_type == InjectionType.SINGLE:
            # Flip a specific bit at position [row_idx, col_idx] for each input in the batch
            # Input shape: [batch_size, row_len, col_len]
            
            original_shape = faulty_output.shape
            
            if len(original_shape) != 3:
                raise ValueError(f"SINGLE injection expects 3D tensor [batch_size, row_len, col_len], got shape {original_shape}")

            batch_size, row_len, col_len = original_shape
            
            # Select position to corrupt (select once and reuse)
            if not hasattr(self, "_bit_flip_row") or self._bit_flip_row is None:
                self._bit_flip_row = np.random.randint(0, row_len - 1)
                self._bit_flip_col = np.random.randint(0, col_len - 1)

            row_idx = self._bit_flip_row
            col_idx = self._bit_flip_col
            
            # Flip the bit at [row_idx, col_idx] for all batch elements
            for b in range(batch_size):
                # Get the element to corrupt
                element = faulty_output[b, row_idx, col_idx].item()
                
                # Convert float to its integer bit representation (float32)
                element_bytes = struct.pack('f', element)
                element_int = struct.unpack('I', element_bytes)[0]

                # Flip the specified bit (self.bit_position)
                bit_mask = 1 << self.bit_position
                corrupted_int = element_int ^ bit_mask
                
                # Convert back to float
                corrupted_bytes = struct.pack('I', corrupted_int)
                corrupted_float = struct.unpack('f', corrupted_bytes)[0]
                
                # Update the tensor
                faulty_output[b, row_idx, col_idx] = corrupted_float

        elif self.injection_type == InjectionType.BULLET_WAKE:
            # first, the same position for every input in the batch
            # Input shape: [batch_size, height, width]
            faulty_output = faulty_output.view(module_output.shape).to(module_output.device)

            if len(faulty_output.shape) != 4:
                raise ValueError(f"BULLET_WAKE injection expects 4D tensor [batch_size, depth, height, width], got shape {faulty_output.shape}")

            batch_size, height, width = faulty_output.shape

            # Select position to corrupt (select once and reuse)
            if not hasattr(self, "_bullet_row") or self._bullet_row is None:
                self._bullet_row = np.random.randint(0, height - 1)
                self._bullet_col = np.random.randint(0, width - 1)

            row_idx = self._bullet_row
            col_idx = self._bullet_col

            for i in range(batch_size):
                # Get the element to corrupt
                element = faulty_output[i, row_idx, col_idx].item()

                # Convert float to its integer bit representation (float32)
                element_bytes = struct.pack('f', element)
                element_int = struct.unpack('I', element_bytes)[0]

                # Flip the specified bit (self.bit_position)
                bit_mask = 1 << self.bit_position
                corrupted_int = element_int ^ bit_mask

                # Convert back to float
                corrupted_bytes = struct.pack('I', corrupted_int)
                corrupted_float = struct.unpack('f', corrupted_bytes)[0]

                # Update the tensor
                faulty_output[i, row_idx, col_idx] = corrupted_float
            
        elif self.injection_type == InjectionType.SINGLE_RANDOM:
            # select a random position for each input in the batch
            # Input shape: [batch_size, height, width]
            faulty_output = faulty_output.view(module_output_shape).to(module_output_device)
            batch_size, height, width = faulty_output.shape

            if not hasattr(self, "_bullet_row") or self._bullet_row is None:
                self._bullet_row = np.random.randint(0, height - 1)
                self._bullet_col = np.random.randint(0, width - 1)

            # check for the faulty value to apply
            # select a random fp32 value (between min_float and max_float)
            if not hasattr(self, "_faulty_value") or self._faulty_value is None:
                self._faulty_value = torch.FloatTensor(1).uniform_(1e-8, 1e2)

            row_idx = self._bullet_row
            col_idx = self._bullet_col

            faulty_output[: , row_idx, col_idx] = self._faulty_value

        # then load model layer bounds to apply range restriction
        # load from npz file, apply 10% tolerance on min/max
        # then clamp faulty values to layer min/max (if outlier, set to min/max)
        if self.rr_mode != RangeRestrictionMode.NONE:
            if self.rr_min_value is None or self.rr_max_value is None:
                layer_bounds = np.load(os.path.join("data/model_layer_bounds", f"{self.model_name}-{self.dataset_name}-fp32-0-layer_bounds.npz"), allow_pickle=True)
                # layer_name = reconstruct_layer_name(self.model_name, self.microop, self.layer_id)

                bounds = layer_bounds[self.layer_full_name].item()

                layer_min, layer_max = bounds["min"], bounds["max"]
                self.rr_min_value = layer_min * 1.1
                self.rr_max_value = layer_max * 1.1

            if self.rr_mode == RangeRestrictionMode.CLAMP:
                faulty_output = torch.clamp(faulty_output, self.rr_min_value, self.rr_max_value)
            elif self.rr_mode == RangeRestrictionMode.TO_ZERO:
                faulty_output[(faulty_output < self.rr_min_value) | (faulty_output > self.rr_max_value)] = 0.0

        # Move the output back to the original device
        faulty_output = faulty_output.view(module_output_shape).to(module_output_device)
        if self.model_name in configs.TEXT_MODELS and isinstance(module_output, tuple):
            return (faulty_output, *module_output[1:])
        else:
            return faulty_output
        




class GetLayerSize:
    """Class to get the size of a layer in a neural network model.
    This class is used to calculate the size of a layer in terms of the number of parameters and input size.
    It is used in the context of fault injection in neural networks.
    """

    def __init__(self, name):
        self.name = name
        self.input_size = 0
        self.microop_size = 0

    def hook_fn_to_get_layer_size(self, module, module_input, module_output) -> None:
        # global _LAYER_TO_HOOK
        global _HOOKABLE_LAYERS
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        self.input_size = sum(p.numel() for p in module_input)
        self.microop_size = layer_num_parameters * self.input_size

        _HOOKABLE_LAYERS.append((module, self.microop_size, self.input_size, self.name))

        # if self.microop_size > _LAYER_TO_HOOK[-1]:
        # _LAYER_TO_HOOK = [module, self.microop_size, self.input_size]


class RangeRestrictionHook:
    """
    Class to restrict the range of values in a tensor.
    This class is used to clamp the values of a tensor to a specified minimum and maximum range.
    It is used in the context of fault injection in neural networks.
    """

    def __init__(self, model_name, min_value, max_value, mode):
        self.model_name = model_name
        self.min_value = min_value
        self.max_value = max_value
        self.mode = mode

    def hook_fn_to_restrict_range(self, module, module_input, module_output) -> None:
        if self.model_name in configs.TEXT_MODELS and isinstance(module_output, tuple):
            output = module_output[0].clone()
        else:
            output = module_output.clone()

        if self.mode == RangeRestrictionMode.CLAMP:
            filtered = torch.clamp(output, self.min_value, self.max_value)
        elif self.mode == RangeRestrictionMode.TO_ZERO:
            output[(output < self.min_value) | (output > self.max_value)] = 0.0
            filtered = output
        else:
            return output
        
        if self.model_name in configs.TEXT_MODELS and isinstance(module_output, tuple):
            return (filtered, *module_output[1:])
        else:
            return filtered

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


# def select_layer(target: LayerChoice) -> torch.nn.Module:
#     """
#     Selects a layer based on the target choice.
#     Args:
#         target (LayerChoice): The target choice for selecting the layer.
#     Returns:
#         torch.nn.Module: The selected layer.
#     """
#     if target == LayerChoice.FIRST:
#         return _HOOKABLE_LAYERS[0][MODULE]
#     elif target == LayerChoice.MIDDLE:
#         return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2][MODULE]
#     elif target == LayerChoice.LAST:
#         return _HOOKABLE_LAYERS[-1][MODULE]
#     elif target == LayerChoice.FIRST_HALF:
#         return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 4][MODULE]
#     elif target == LayerChoice.MIDDLE_HALF:
#         return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2 + len(_HOOKABLE_LAYERS) // 4][MODULE]
#     elif target == LayerChoice.BEFORE_LAST:
#         return _HOOKABLE_LAYERS[-2][MODULE]
#     else:
#         return ValueError("Invalid layer choice.")
    

def select_layer(target: int) -> torch.nn.Module:
    """
    Selects a layer based on the target choice.
    Args:
        target (LayerChoice): The target choice for selecting the layer.
    Returns:
        torch.nn.Module: The selected layer.
    """
    if target >= 0 and target < len(_HOOKABLE_LAYERS):
        return _HOOKABLE_LAYERS[target][MODULE], _HOOKABLE_LAYERS[target][LAYER_FULL_NAME]
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
    seed=0,
    bit_position=DEFAULT_BIT_POSITION,
    dataset_name=None,
    range_restriction_mode=RangeRestrictionMode.NONE,
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
    handlers = list()
    hookable_indices = list()

    range_restrict_handlers = list()

    for layer_id, (name, layer) in enumerate(model.named_modules()):
        # print(layer.__class__.__name__.strip())
        if model_name == configs.FACEBOOK_BART_DASH and ".fc2" in name:
            layer.__class__.__name__ = configs.BART_MLP

        if layer.__class__.__name__.strip() == microop:
            # layers.append((layer, layer_id))
            hook = GetLayerSize(name)
            handler = layer.register_forward_hook(hook.hook_fn_to_get_layer_size)
            handlers.append(handler)
            hookable_indices.append(layer_id)

    if model_name in configs.TEXT_MODELS:
        _ = model(**dummy_input)
    else:
        _ = model(dummy_input)

    for handler in handlers:
        handler.remove()

    layer, name = select_layer(target)

    hook = MicroopHook(
        model_name,
        microop,
        batch_size,
        nb_inputs,
        target,
        fault_model,
        injection_type,
        layer_full_name=name,
        seed=seed,
        bit_position=bit_position,
        dataset_name=dataset_name,
        rr_mode=range_restriction_mode,
    )
    handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)

    # protect every layer after the injection layer with range restriction
    if range_restriction_mode != RangeRestrictionMode.NONE:
        # load from npz file, apply 10% tolerance on min/max
        layer_bounds = np.load(os.path.join("data/model_layer_bounds", f"{model_name}-{dataset_name}-fp32-0-layer_bounds.npz"), allow_pickle=True)

        for layer_id, (name, layer) in enumerate(model.named_modules()):
            if layer_id <= hookable_indices[target]:
                continue

            if name not in layer_bounds:
                continue

            bounds = layer_bounds[name].item()
            layer_min, layer_max = bounds["min"], bounds["max"]

            layer_min = layer_min * 1.1
            layer_max = layer_max * 1.1
            range_hook = RangeRestrictionHook(model_name, layer_min, layer_max, range_restriction_mode)
            range_handler = layer.register_forward_hook(range_hook.hook_fn_to_restrict_range)
            range_restrict_handlers.append(range_handler)

    return hook, handler, range_restrict_handlers, 


def run_inference(model, images, device):
    with torch.no_grad():
        output = model(images)
        if "cuda" in device:
            torch.cuda.synchronize()
        out_top_k = get_top_k_labels(output, configs.TOP_1)
        out_top_k_prob = get_top_k_probs(output, configs.TOP_2)
        return out_top_k, out_top_k_prob
    

def run_inference_gpt2(model, inputs, device):
    with torch.no_grad():
        output = model(**inputs)
        if "cuda" in device:
            torch.cuda.synchronize()
        logits = output.logits

        out_top_k = get_top_k_labels(logits, configs.TOP_1, dim=-1)
        out_top_k_prob = get_top_k_probs(logits, configs.TOP_2, dim=-1)
        return out_top_k, out_top_k_prob

def reconstruct_layer_name(model_name, microop, layer_id):
    if model_name == configs.GPT2:
        base = configs.LAYER_NAMES_MAPPING[model_name][configs.GPT2_BLOCK]
    elif model_name == configs.VIT_BASE_PATCH16_224:
        base = configs.LAYER_NAMES_MAPPING[model_name][configs.BLOCK]
    else:
        raise ValueError(f"Model {model_name} not supported for layer name reconstruction.")
    
    base += str(layer_id)

    if not microop in [configs.BLOCK, configs.GPT2_BLOCK]:
        base += configs.LAYER_NAMES_MAPPING[model_name][microop]

    return base