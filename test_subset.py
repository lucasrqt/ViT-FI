#! /usr/bin/env python3

import torch
from torch.utils.data import DataLoader, Subset
import argparse
import pandas as pd
import model_utils
import configs
import time


def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser(description="Subset of Imagenet dataset.", add_help=True)
    parser.add_argument("--model", type=str, default=configs.VIT_BASE_PATCH16_224, help="Model name.", choices=[configs.VIT_BASE_PATCH16_224, configs.SWIN_BASE_PATCH4_WINDOW7_224])
    parser.add_argument("--dataset", type=str, default=configs.IMAGENET, help="Dataset name.", choices=[configs.IMAGENET])
    parser.add_argument("--precision", type=str, default=configs.FP32, help="Precision.", choices=[configs.FP32])
    parser.add_argument("--top-diff-th", type=float, default=0.1, help="Top1-top2 difference threshold.")
    parser.add_argument("--batchsize", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-batches", type=int, default=4, help="Number of batches.")
    return parser.parse_args()


def select_subset(test_set, top_diff_th, file_probabilities, num_inputs) -> Subset:
    df = pd.read_csv(file_probabilities, index_col=0)
    df = df.sort_values(by="top_diff", ascending=True)
    indices = df.index.tolist()[:num_inputs]
    return indices, Subset(test_set, indices)


def main() -> None:
    args = parse_args()
    model_name = args.model
    dataset_name = args.dataset
    precision = args.precision
    top_diff_th = args.top_diff_th
    batch_size = args.batchsize
    num_batches = args.num_batches

    # model = model_utils.get_model(model_name, configs.FP32)
    transforms = model_utils.get_vit_transforms(model_name, configs.FP32)
    test_set, _ = model_utils.get_dataset(dataset_name, transforms, batch_size)

    print(f"Loading subset of {dataset_name} dataset.")
    start = time.time()
    corr_pred_indices, correct_predictions = model_utils.get_correct_indices(test_set, f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv")

    # _ , subset = model_utils.get_correct_indices(test_set, f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv")
    indices, subset = select_subset(correct_predictions, top_diff_th, f"data/{model_name}_{dataset_name}_{precision}_top5_prob.csv", batch_size * num_batches)
    stop = time.time()
    print(f"Subset loaded with {len(indices)} images in {stop - start}s.")
    print(indices)
    df_test = pd.read_csv(f"data/{model_name}_{dataset_name}_{precision}_top5_prob.csv", index_col=0)
    df_correct = pd.read_csv(f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv")
    # print(df_correct.loc[indices])
    
    # print(df_test.loc[indices])

    data_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    for i, (images, labels) in enumerate(data_loader):
        print(f"Batch {i}")
        print(labels)



if __name__ == "__main__":
    main()