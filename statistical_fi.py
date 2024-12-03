import torch
import configs
import random

def select_microop(model_name):
    if model_name in configs.SWIN_MODELS:
        return random.choice(configs.SWIN_MICROOPS)
    elif model_name in configs.CLASSICAL_VIT_MODELS:
        return random.choice(configs.VIT_MICROOPS)
    else:
        return ValueError(f"Model {model_name} not supported.")

def hook_microop(model, microop):
    pass

def run_with_fault(model, images, microop):
    # TODO: hook the microop and select one layer of this microop to inject the fault

    output = model(images)
    return output
    

def run_without_fault(model, images):
    output = model(images)
    return output


