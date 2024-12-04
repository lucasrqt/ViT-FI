#! /usr/bin/env python3

import configs
import statistical_fi
import argparse
import model_utils
import result_data_utils
import torch
from compare_utils import get_top_k_labels
import random

class LayerHook:
    def __init__(self, microop, fault_model):
        self.microop = microop

    def hook_function_to_extract_layers(self, module, module_input, kwargs, module_output) -> None:
            layer_class = module.__class__.__name__.strip()
            layer_num_parameters = sum(p.numel() for p in module.parameters())
            save_path = f"id_{1}_name_{0}_class_{layer_class}"
            print("HOOK")
            print(module_input[0])
            faulty_input = torch.mul(module_input[0], 0.14)
            module_input = (faulty_input,)
            print(module_input)

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


def run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, result_df, result_file) -> None:
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()
        
        images = images.to(device)
        labels = labels.to(device)

        # microop = statistical_fi.select_microop(model_name)
        out_wo_fault = statistical_fi.run_inference(model, images, device)
        out_with_fault = statistical_fi.run_inference(model_for_fault, images, device)

        print("-" * 80)
        print(f" [+] Image {i} - Microop: {microop}")
        print(f" [+] Ground truth: {labels} - Prediction without fault: {out_wo_fault} - Prediction with fault: {out_with_fault}")

        result_df = result_data_utils.append_row(result_df, model_name, dataset_name, precision, microop, labels, out_wo_fault, out_with_fault)
        result_data_utils.save_result_data(result_df, configs.RESULTS_DIR, result_file)


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
    fault_model_threshold = f"{args.fault_model_threshold:.2e}"
    microop = args.microop
    if microop is None:
        raise ValueError("Microoperation not defined.")

    if statistical_fi.check_microop(model_name, microop) is False:
        raise ValueError(f"Microoperation {microop} not supported by the model {model_name}.")

    print(" [+] Model init...")
    model = model_utils.get_model(model_name, precision)
    model_for_fault = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)

    fault_model = statistical_fi.get_fault_model(configs.FAULT_MODEL_FILE, model_name, microop, precision, fault_model_threshold)
    if fault_model.empty:
        raise ValueError("Fault model not found.")
    statistical_fi.hook_microop(model_for_fault, microop, fault_model)

    data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)
    
    result_file = result_data_utils.get_result_filename(model_name, dataset_name, precision, fault_model_threshold)
    result_df = result_data_utils.init_result_data(configs.RESULTS_DIR, result_file, configs.RESULT_COLUMS)

    print(" [+] Running injections...")
    run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, result_df)



if __name__ == "__main__":
    main()
