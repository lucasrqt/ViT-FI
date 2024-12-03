#! /usr/bin/env python3

import configs
import statistical_fi
import argparse
import model_utils
import result_data_utils

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform high-level fault injections on ViT model according neutron beam fault model.", add_help=True)
    parser.add_argument("-m", "--model", type=str, default=configs.VIT_BASE_PATCH16_224, help="Model name.", choices=configs.VIT_CLASSIFICATION_CONFIGS)
    parser.add_argument("-d", "--dataset", type=str, default=configs.IMAGENET, help="Dataset name.", choices=[configs.IMAGENET, configs.COCO, configs.CIFAR10])
    parser.add_argument("-b", "--batch-size", type=int, default=configs.DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("-p", "--precision", type=str, default=configs.FP32, help="Precision of the model and inputs.", choices=[configs.FP16, configs.FP32])
    parser.add_argument("-d", "--device", type=str, default=configs.GPU_DEVICE, help="Device to run the model.", choices=[configs.CPU, configs.GPU_DEVICE])
    parser.add_argument("-s", "--seed", type=int, default=configs.TORCH_SEED, help="Random seed.")
    parser.add_argument("--fault-model-threshold", type=float, default=1e-03, help="Threshold for the fault model data.")
    parser.add_argument("--result-file", type=str, default=configs.RESULTS_FILE, help=f"File to store the results. File will be saved in {configs.RESULTS_DIR}.")
    return parser.parse_args()


def run_injections(model_name, dataset_name, model, data_loader, precision, device, result_df, result_file) -> None:
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()
        
        images = images.to(device)
        labels = labels.to(device)

        microop = statistical_fi.select_microop(model_name)
        out_wo_fault = statistical_fi.run_without_fault(model, images)
        out_with_fault = statistical_fi.run_with_fault(model, images, microop)

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

    print(" [+] Model init...")
    model = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)


    data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)
    
    result_file = result_data_utils.get_result_filename(model_name, dataset_name, precision, fault_model_threshold)
    result_df = result_data_utils.init_result_data(configs.RESULTS_DIR, result_file, configs.RESULT_COLUMS)

    print(" [+] Running injections...")
    run_injections(model_name, dataset_name, model, data_loader, precision, device, result_df)



if __name__ == "__main__":
    main()
