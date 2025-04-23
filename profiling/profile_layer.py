#! /usr/bin/env python3

import os
import sys

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join("..", "utils")))

import configs
import model_utils
import torch
from statistical_fi import LayerChoice
import pandas as pd
from functools import partial
import cli.logger_formatter as logger_formatter

MODELS = [
    configs.VIT_BASE_PATCH16_224,
    configs.SWIN_BASE_PATCH4_WINDOW7_224,
]

MICROOPS = {
    configs.VIT_BASE_PATCH16_224: configs.VIT_MODULES,
    configs.SWIN_BASE_PATCH4_WINDOW7_224: configs.SWIN_MODULES,
}

BATCH_SIZE = 32
DATASET = "imagenet"

NAME, LAYER_ID, MODULE, MICROOP_SIZE, INPUT_SIZE = 0, 1, 2, 3, 4

_HOOKABLE_LAYERS = list()
_SMALLEST = [1e30]
_LARGEST = [1e-30]

DATA_RESULT = "data/layer_data.csv"


class GetLayerSize:
    def __init__(self):
        self.input_size = 0
        self.microop_size = 0

    def hook_fn_to_get_layer_size(
        self, module, module_input, module_output, name, layer_id
    ) -> None:
        # global _LAYER_TO_HOOK
        global _HOOKABLE_LAYERS, _SMALLEST, _LARGEST
        layer_num_parameters = sum(p.numel() for p in module.parameters())
        self.input_size = sum(p.numel() for p in module_input)
        self.microop_size = layer_num_parameters * self.input_size

        _HOOKABLE_LAYERS.append(
            (name, layer_id, module, self.microop_size, self.input_size)
        )

        if self.microop_size > _LARGEST[-1]:
            _LARGEST = [name, layer_id, module, self.microop_size, self.input_size]

        if self.microop_size < _SMALLEST[-1]:
            _SMALLEST = [name, layer_id, module, self.microop_size, self.input_size]


def hook_microop(
    model, model_name, microop, batch_size, dummy_input
) -> torch.utils.hooks.RemovableHandle:
    handlers = list()
    for layer_id, (name, layer) in enumerate(model.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            # layers.append((layer, layer_id))
            hook = GetLayerSize()
            hook_w_params = partial(
                hook.hook_fn_to_get_layer_size, name=name, layer_id=layer_id
            )
            handler = layer.register_forward_hook(hook_w_params)
            handlers.append(handler)

    _ = model(dummy_input)

    for handler in handlers:
        handler.remove()


def main() -> None:

    logger = logger_formatter.logging_setup(__name__, None, False)

    layer_data = []

    for model_name in MODELS:
        model = model_utils.get_model(model_name, configs.FP32)
        transforms = model_utils.get_vit_transforms(model, configs.FP32)
        _, test_loader = model_utils.get_dataset(DATASET, transforms, BATCH_SIZE)
        model = model.to(configs.GPU_DEVICE)

        logger.info(f"Model {model_name} loaded.")

        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(configs.GPU_DEVICE)

        logger.info("Input loaded.")

        for microop in MICROOPS[model_name]:
            logger.info(f"Processing {model_name} with microop {microop}.")
            global _HOOKABLE_LAYERS, _SMALLEST, _LARGEST
            _HOOKABLE_LAYERS = list()
            _SMALLEST = [1e30]
            _LARGEST = [1e-30]
            hook_microop(model, model_name, microop, BATCH_SIZE, inputs)

            data = {
                "model": model_name,
                "microop": microop,
                f"{LayerChoice.FIRST}": _HOOKABLE_LAYERS[0][NAME],
                f"{LayerChoice.LAST}": _HOOKABLE_LAYERS[-1][NAME],
                f"{LayerChoice.MIDDLE}": _HOOKABLE_LAYERS[len(_HOOKABLE_LAYERS) // 2][
                    NAME
                ],
                f"{LayerChoice.SMALLEST}": _SMALLEST[NAME],
                f"{LayerChoice.LARGEST}": _LARGEST[NAME],
            }

            # sorted_layers = sorted(_HOOKABLE_LAYERS, key=lambda x: x[MICROOP_SIZE])

            # data[f"{LayerChoice.SMALLEST}"] = sorted_layers[0][NAME]
            # data[f"{LayerChoice.LARGEST}"] = sorted_layers[-1][NAME]

            layer_data.append(data)

    data_df = pd.DataFrame(layer_data)
    data_df.to_csv(DATA_RESULT, index=False)


if __name__ == "__main__":
    main()
