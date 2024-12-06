#! /usr/bin/env python3

import configs
import statistical_fi
import argparse
import model_utils
import result_data_utils
import torch
import pandas as pd
import torchvision.datasets as tv_datasets
import os
import numpy as np

class MicroopHook():
    def __init__(self, corrupted_output):
        self.corrupted_output = corrupted_output
        self.injected = False

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        if module_output.shape != self.corrupted_output.shape:
            return module_output

        self.injected = True
        return self.corrupted_output
    
class ShapeHook():
    def __init__(self):
        self.shape = None

    def hook_fn_to_inject_fault(self, module, module_input, module_output) -> None:
        self.shape = module_output.shape

        return module_output

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform high-level fault injections on ViT model according neutron beam fault model.", add_help=True)
    parser.add_argument("-m", "--model", type=str, default=configs.VIT_BASE_PATCH16_224, help="Model name.", choices=configs.VIT_CLASSIFICATION_CONFIGS)
    parser.add_argument("-D", "--dataset", type=str, default=configs.IMAGENET, help="Dataset name.", choices=[configs.IMAGENET, configs.COCO, configs.CIFAR10])
    parser.add_argument("-b", "--batch-size", type=int, default=configs.DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("-p", "--precision", type=str, default=configs.FP32, help="Precision of the model and inputs.", choices=[configs.FP16, configs.FP32])
    parser.add_argument("-d", "--device", type=str, default=configs.GPU_DEVICE, help="Device to run the model.", choices=[configs.CPU, configs.GPU_DEVICE])
    parser.add_argument("-M", "--microop", type=str, default=None, help="Microoperation to inject the fault.", choices=configs.MICROBENCHMARK_MODULES)
    parser.add_argument("-s", "--seed", type=int, default=configs.TORCH_SEED, help="Random seed.")
    parser.add_argument("--fault-model-threshold", type=float, default=1e-03, help="Threshold for the fault model data.")
    return parser.parse_args()


def run_injections(model, model_for_fault, images, labels, device, hook) -> None:
    model.eval()
    model.to(device)

    model_for_fault.eval()
    model_for_fault.to(device)
        
    images = images.to(device)
    labels = labels.to(device)

    # microop = statistical_fi.select_microop(model_name)
    out_wo_fault = statistical_fi.run_inference(model, images, device).squeeze()
    out_with_fault = statistical_fi.run_inference(model_for_fault, images, device).squeeze()
    labels = labels

    print("-" * 80)
    for j in range(len(images)):
        print(f" [+] Image {j+1} - Ground truth: {labels[j]} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}")


def main() -> None:
    args = parse_args()

    # Parse arguments
    print(" [+] Parsing arguments...")
    precision = args.precision
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    model_name = args.model
    dataset_name = args.dataset
    fault_model_threshold = f"{args.fault_model_threshold:.0e}"
    microop = args.microop
    if microop is None:
        raise ValueError("Microoperation not defined.")

    if statistical_fi.check_microop(model_name, microop) is False:
        raise ValueError(f"Microoperation {microop} not supported by the model {model_name}.")

    print(" [+] Model init...")
    model = model_utils.get_model(model_name, precision)
    model_for_fault = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)

    #### TEST CASE
    # dummy_input = torch.randn(32, 3, 224, 224)
    # out_wo_fault = statistical_fi.run_inference(model, dummy_input, device).squeeze()
    # out_with_fault = statistical_fi.run_inference(model_for_fault, dummy_input, device).squeeze()

    # print("-" * 80)
    # print(f" [+] Batch {0} - Microop: {microop}")
    # for j in range(len(dummy_input)):
    #     print(f" [+] Image {j+1} - Ground truth: {out_wo_fault[j].item()} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}")
    ####

    # data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)
    # Set a sampler on the CPU
    sampler_generator = torch.Generator(device=configs.CPU)
    sampler_generator.manual_seed(configs.TORCH_SEED)

    test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transforms,
                                                split='val')
    subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=batch_size,
                                            generator=sampler_generator)
    data_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True)

    images, labels = next(iter(data_loader))
    

    config = f"{model_name}_{microop}_{precision}_{fault_model_threshold}"
    print(config)

    microbench_outputs = [file for file in os.listdir("data/microbench/") if config in file]
    print(f" [+] Found {len(microbench_outputs)} outputs for {config}.")

    layers = list()
    for layer_id, (name, layer) in enumerate(model_for_fault.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            layers.append(layer)

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    shapes = list()
    for layer in layers:
        shape_hook = ShapeHook()
        handler = layer.register_forward_hook(shape_hook.hook_fn_to_inject_fault)
        model_for_fault(dummy_input)
        shapes.append(shape_hook.shape)
        handler.remove()
    
    print(" [+] Got shapes.")

    for layer_id, (layer, shape) in enumerate(zip(layers, shapes)):
        print(f" [+] Injecting fault in layer {layer_id} for microop {microop}...")
        for output in microbench_outputs:
            corrupted_output = np.load(f"data/microbench/{output}")
            corrupted_output = torch.tensor(corrupted_output["alt_output"]).to(device)
            print(corrupted_output.shape , shape)
            if corrupted_output.shape != shape:
                continue

            hook = MicroopHook(corrupted_output)
            handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)
            run_injections(model, model_for_fault, images, labels, device, hook)
            handler.remove()


        # print(" [+] Running injections...")
        # run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, result_df, result_file)


def main2() -> None:
    args = parse_args()

    # Parse arguments
    print(" [+] Parsing arguments...")
    precision = args.precision
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    model_name = args.model
    dataset_name = args.dataset
    fault_model_threshold = f"{args.fault_model_threshold:.0e}"
    microop = args.microop
    if microop is None:
        raise ValueError("Microoperation not defined.")

    if statistical_fi.check_microop(model_name, microop) is False:
        raise ValueError(f"Microoperation {microop} not supported by the model {model_name}.")

    print(" [+] Model init...")
    model = model_utils.get_model(model_name, precision)
    model_for_fault = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)

    #### TEST CASE
    # dummy_input = torch.randn(32, 3, 224, 224)
    # out_wo_fault = statistical_fi.run_inference(model, dummy_input, device).squeeze()
    # out_with_fault = statistical_fi.run_inference(model_for_fault, dummy_input, device).squeeze()

    # print("-" * 80)
    # print(f" [+] Batch {0} - Microop: {microop}")
    # for j in range(len(dummy_input)):
    #     print(f" [+] Image {j+1} - Ground truth: {out_wo_fault[j].item()} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}")
    ####

    # data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)
    # Set a sampler on the CPU
    sampler_generator = torch.Generator(device=configs.CPU)
    sampler_generator.manual_seed(configs.TORCH_SEED)

    test_set = tv_datasets.imagenet.ImageNet(root=configs.IMAGENET_DATASET_DIR, transform=transforms,
                                                split='val')
    subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=batch_size,
                                            generator=sampler_generator)
    data_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
                                                shuffle=False, pin_memory=True)

    images, labels = next(iter(data_loader))
    

    config = f"{model_name}_{microop}_{precision}_{fault_model_threshold}"
    print(config)

    microbench_outputs = [file for file in os.listdir("data/microbench/") if config in file]
    print(f" [+] Found {len(microbench_outputs)} outputs for {config}.")

    layers = list()
    for layer_id, (name, layer) in enumerate(model_for_fault.named_modules()):
        if layer.__class__.__name__.strip() == microop:
            layers.append(layer)

    # dummy_input = torch.randn(batch_size, 3, 224, 224)
    # shapes = list()
    # for layer in layers:
    #     shape_hook = ShapeHook()
    #     handler = layer.register_forward_hook(shape_hook.hook_fn_to_inject_fault)
    #     model_for_fault(dummy_input)
    #     shapes.append(shape_hook.shape)
    #     handler.remove()
    
    # print(" [+] Got shapes.")

    fault_model = statistical_fi.get_fault_model(configs.FAULT_MODEL_FILE, model_name, microop, precision, fault_model_threshold)
    if fault_model.empty:
        raise ValueError("Fault model not found.")

    for layer_id, layer in enumerate(layers):
        print(f" [+] Injecting fault in layer {layer_id} for microop {microop}...")
        hook = statistical_fi.MicroopHook(microop, layer_id, fault_model)
        handler = layer.register_forward_hook(hook.hook_fn_to_inject_fault)
        run_injections(model, model_for_fault, images, labels, device, hook)
        handler.remove()


        # print(" [+] Running injections...")
        # run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, result_df, result_file)


if __name__ == "__main__":
    main2()
