#! /usr/bin/env python3

import configs
import statistical_fi
import argparse
import model_utils
import result_data_utils
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform high-level fault injections on ViT model according neutron beam fault model.", add_help=True)
    parser.add_argument("-m", "--model", type=str, default=configs.VIT_BASE_PATCH16_224, help="Model name.", choices=configs.VIT_CLASSIFICATION_CONFIGS)
    parser.add_argument("-D", "--dataset", type=str, default=configs.IMAGENET, help="Dataset name.", choices=[configs.IMAGENET, configs.COCO, configs.CIFAR10])
    parser.add_argument("-b", "--batch-size", type=int, default=configs.DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("-p", "--precision", type=str, default=configs.FP32, help="Precision of the model and inputs.", choices=[configs.FP16, configs.FP32])
    parser.add_argument("-d", "--device", type=str, default=configs.GPU_DEVICE, help="Device to run the model.", choices=[configs.CPU, configs.GPU_DEVICE])
    parser.add_argument("-M", "--microop", type=str, default=None, help="Microoperation to inject the fault.", choices=configs.MICROBENCHMARK_MODULES)
    parser.add_argument("-s", "--seed", type=int, default=configs.SEED, help="Random seed.")
    parser.add_argument("--fault-model-threshold", type=float, default=1e-03, help="Threshold for the fault model data.")
    parser.add_argument("--inject-on-correct-predictions", action="store_true", help="Inject faults only on correct predictions.", default=False)
    parser.add_argument("--load-critical", action="store_true", help="Only load the images that are critical for the fault injection.", default=False)
    parser.add_argument("--save-critical-logits", action="store_true", help="Save the logits of the critical images.", default=False)
    parser.add_argument("--save-top5prob", action="store_true", help="Save the top 5 probabilities of the critical images.", default=False)
    return parser.parse_args()


def run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, batch_size, result_df, result_file) -> None:
    model.eval()
    model.to(device)

    model_for_fault.eval()
    model_for_fault.to(device)

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()
        
        images = images.to(device)
        labels = labels.to(device)

        # microop = statistical_fi.select_microop(model_name)
        out_wo_fault = statistical_fi.run_inference(model, images, device).squeeze()
        out_with_fault = statistical_fi.run_inference(model_for_fault, images, device).squeeze()

        print("-" * 80)
        print(f" [+] Batch {i} - Microop: {microop}")
        for j in range(len(images)):
            print(f" [+] Image {(i*batch_size)+j+1} - Ground truth: {labels[j]} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}")

            result_df = result_data_utils.append_row(result_df, model_name, dataset_name, precision, microop, labels[j].item(), out_wo_fault[j].item(), out_with_fault[j].item())
            result_data_utils.save_result_data(pd.DataFrame(result_df), configs.RESULTS_DIR, result_file)

    print(" [+] Done.")


def get_faulty_top5(model_name, microop, model, data_loader, precision, device, batch_size) -> None:
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()
        
        images = images.to(device)
        labels = labels.to(device)

        # microop = statistical_fi.select_microop(model_name)
        with torch.no_grad():
            out_with_fault = model(images)
            labels = labels
            if "cuda" in device:
                torch.cuda.synchronize()

            print("-" * 80)
            print(f" [+] Batch {i} - Microop: {microop}")
            
            top5prob = torch.nn.functional.softmax(out_with_fault, dim=1)
            top5prob = top5prob.cpu()
            top5prob = torch.topk(top5prob, k=5)
            for j in range(len(images)):
                path = f"data/top5prob/faulty-{model_name}-{microop}-top5prob_{(i*batch_size)+j}.pt"
                tensor = torch.cat((top5prob.indices[j].unsqueeze(0), top5prob.values[j].unsqueeze(0)), dim=0)
                torch.save(tensor, path)
                print(f" [+] Image {(i*batch_size)+j+1} saved.")


    print(" [+] Done.")


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
    inject_on_corr_preds = args.inject_on_correct_predictions
    save_critical_logits = args.save_critical_logits
    save_top5prob = args.save_top5prob

    np.random.seed(seed)
    torch.manual_seed(seed)

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

    test_set, data_loader = model_utils.get_dataset(dataset_name, transforms, batch_size)
    if inject_on_corr_preds:
        _ , subset = model_utils.get_correct_indices(test_set, f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv")
        if args.load_critical:
            df = pd.read_csv(f"data/fi_critical_images.csv")
            df = df[(df["model"] == model_name) & (df["microop"] == microop)]
            if df.empty:
                raise ValueError("No critical images found.")
            indices = df["image_id"].tolist()
            # full_batchs = []
            batch_indices = []
            for index in indices:
                batch_id = model_utils.get_batch_id(index, batch_size)
                batch_indices.append(batch_id)
            #     full_batchs += range(batch_id*batch_size, (batch_id+1)*batch_size)
            # subset = Subset(subset, full_batchs)

        print(f" [+] {len(subset)} correct predictions found.")
        data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        print(f" [+] Injecting faults on correct predictions only.")

    dummy_input, _ = next(iter(data_loader))
    fault_model = statistical_fi.get_fault_model(configs.FAULT_MODEL_FILE, model_name, microop, precision, fault_model_threshold)
    if fault_model.empty:
        raise ValueError("Fault model not found.")
    hook, handler = statistical_fi.hook_microop(model_for_fault, model_name, microop, batch_size, fault_model, dummy_input)
    if args.load_critical:
        hook.set_critical_batches(batch_indices)
        hook.set_save_critical_logits(save_critical_logits)
    del dummy_input
    
    print(f" [+] Injecting on {len(data_loader)} batches of size {batch_size}...")

    result_file = result_data_utils.get_result_filename(model_name, dataset_name, precision, microop, fault_model_threshold)
    result_df = result_data_utils.init_result_data(configs.RESULTS_DIR, result_file, configs.RESULT_COLUMS)

    print(" [+] Running injections...")
    if save_top5prob:
        get_faulty_top5(model_name, microop, model_for_fault, data_loader, precision, device, batch_size)
    else:
        run_injections(model_name, dataset_name, microop, model, model_for_fault, data_loader, precision, device, batch_size, result_df, result_file)

    handler.remove()



if __name__ == "__main__":
    main()
