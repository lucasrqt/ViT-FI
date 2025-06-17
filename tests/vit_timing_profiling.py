#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import pandas as pd
import enum
import argparse

import torch.utils.data

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join("..", "utils")))

import configs
import cli.logger_formatter as logger_formatter
import utils.model_utils as model_utils
import utils.result_data_utils as result_data_utils

_HOOKABLE_LAYERS = []
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

class GetLayerSize:
    def __init__(self):
        self.input_size = 0
        self.microop_size = 0

    def hook_fn_to_get_layer_size(self, module, module_input, module_output):
        global _HOOKABLE_LAYERS
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        self.input_size = sum(p.numel() for p in module_input)
        self.microop_size = layer_num_parameters * self.input_size
        _HOOKABLE_LAYERS.append((module, self.microop_size, self.input_size))

class LayerHook:
    def __init__(self, name="layer"):
        self.name = name
        self.start_event = None
        self.end_event = None
        self.hook_handle_pre = None
        self.hook_handle_post = None
        self.times = []

    def _pre_hook(self, module, input):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time_cpu = torch.perf_counter()

    def _post_hook(self, module, input, output):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            elapsed = self.start_event.elapsed_time(self.end_event)  # ms
        else:
            elapsed = (torch.perf_counter() - self.start_time_cpu) * 1000  # ms
        self.times.append(elapsed)

    def remove(self):
        if self.hook_handle_pre:
            self.hook_handle_pre.remove()
        if self.hook_handle_post:
            self.hook_handle_post.remove()


def select_layer(target: LayerChoice) -> torch.nn.Module:
    if target == LayerChoice.FIRST:
        return _HOOKABLE_LAYERS[0][MODULE]
    elif target == LayerChoice.MIDDLE:
        return _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2][MODULE]
    elif target == LayerChoice.LAST:
        return _HOOKABLE_LAYERS[-1][MODULE]
    else:
        raise ValueError("Invalid layer choice.")


def run_profiling(model_name, precision, batch_size, device, dataset_name, microop, target_layer, logger, num_steps, warmup_steps):
    global _HOOKABLE_LAYERS

    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)

    logger.info("Model init...")
    model = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)
    test_set, data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)

    logger.info("Dataset loaded.")

    handlers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            # layers.append((layer, layer_id))
            hook = GetLayerSize()
            handler = layer.register_forward_hook(hook.hook_fn_to_get_layer_size)
            handlers.append(handler)

    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            break

    for handler in handlers:
        handler.remove()

    if not _HOOKABLE_LAYERS:
        raise RuntimeError("No hookable layers found matching the microop.")
    
    logger.info(f"Hookable layers found: {len(_HOOKABLE_LAYERS)}")

    # Warmup
    for i, (inputs, _) in enumerate(data_loader):
        if i >= warmup_steps:
            logger.info(f"Warmup completed after {i} steps.")
            break
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)

    # Full model timing before hook timing
    full_model_times = []
    for i, (inputs, _) in enumerate(data_loader):
        if i >= num_steps:
            logger.info(f"Full model timing completed after {i} steps.")
            break

        inputs = inputs.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(inputs)
            end_event.record()
            torch.cuda.synchronize()
            full_model_times.append(start_event.elapsed_time(end_event))
        else:
            start_time = torch.perf_counter()
            with torch.no_grad():
                _ = model(inputs)
            full_model_times.append((torch.perf_counter() - start_time) * 1000)

    selected_layer = select_layer(target_layer)

    hook = LayerHook(name=str(target_layer))
    hook.hook_handle_pre = selected_layer.register_forward_pre_hook(hook._pre_hook)
    hook.hook_handle_post = selected_layer.register_forward_hook(hook._post_hook)

    logger.info("Starting inference timing...")

    # Hooked layer timing
    for i, (inputs, _) in enumerate(data_loader):
        if i >= num_steps:
            logger.info(f"Hooked layer timing completed after {i} steps.")
            break
        inputs = inputs.to(device)
        with torch.no_grad():
            _ = model(inputs)

    hook.remove()
    _HOOKABLE_LAYERS = list()
    # Save to CSV
    return {
        "avg_layer_time_ms": np.mean(hook.times),
        "avg_full_model_time_ms": np.mean(full_model_times)
    }
    # df.to_csv("layer_timing_results.csv", index=False)
    # print(df.describe())


def main():
    parser = argparse.ArgumentParser(description="Run layer timing profiling for ViT models.", add_help=True)
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging.")
    parser.add_argument("--num-steps", type=int, default=200, help="Number of steps for profiling.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Number of warmup steps before profiling.")
    args = parser.parse_args()

    num_steps = args.num_steps
    warmup_steps = args.warmup_steps

    models = [
        configs.VIT_BASE_PATCH16_224,
        configs.SWIN_BASE_PATCH4_WINDOW7_224,
    ]

    dataset = configs.IMAGENET
    precisions = [
        configs.FP32, 
        # configs.FP16,
    ]

    batch_size = configs.DEFAULT_BATCH_SIZE
    devices = [
        configs.GPU_DEVICE,
        # configs.CPU,
    ]
    layers = [
        LayerChoice.FIRST,
        # LayerChoice.MIDDLE,
        LayerChoice.LAST,
    ]

    microops_per_model = {
        configs.VIT_BASE_PATCH16_224: configs.VIT_MODULES,
        configs.SWIN_BASE_PATCH4_WINDOW7_224: configs.SWIN_MODULES,
    }

    verbose = False
    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    datas = []

    for model in models:
        for precision in precisions:
            for device in devices:
                for layer in layers:
                    microops = microops_per_model.get(model, configs.MICROBENCHMARK_MODULES)
                    for microop in microops:
                        logger.info(f"Profiling {model} with {precision} on {device} for {microop} at layer {layer}")
                        data = run_profiling(model, precision, batch_size, device, dataset, microop, layer, logger, num_steps, warmup_steps)
                        datas.append({
                            "model": model,
                            "microop": microop,
                            "layer": str(layer),
                            **data
                        })

    df = pd.DataFrame(datas)
    results_file = "layer_timing_results.csv"
    df.to_csv(results_file, index=False)
    print(df.describe())
    logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
