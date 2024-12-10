#! /usr/bin/env python3

import torch
import argparse
import model_utils
import configs
import logging

BATCH_SIZE = 32
INDICES_IDX, VALUES_IDX = 0, 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Get top5 accuracy', add_help=True)
    parser.add_argument('--model', type=str, help='Model file')
    parser.add_argument('--precision', type=str, default=configs.FP32, help='Precision of the model')
    parser.add_argument('--dataset', type=str, default=configs.IMAGENET, help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--device', type=str, default=configs.GPU_DEVICE, help='Device to use', choices=[configs.GPU_DEVICE, configs.CPU])
    return parser.parse_args()


def save_top5_prob(output, path) -> None:
    top5prob = torch.nn.functional.softmax(output, dim=1)
    top5prob = top5prob.cpu()
    top5prob = torch.topk(top5prob, k=5)
    torch.save(top5prob, path)


def main() -> None:
    logger = logging.getLogger(__name__)

    args = parse_args()
    model_name = args.model
    precision = args.precision
    dataset = args.dataset
    batch_size = args.batch_size
    device = args.device

    model = model_utils.get_model(model_name, precision)
    transforms = model_utils.get_vit_transforms(model, precision)
    test_set, test_loader = model_utils.get_dataset(dataset, transforms, batch_size)

    indices, subset = model_utils.get_correct_indices(test_set, f"data/{model_name}_{dataset}_{precision}_correct_predictions.csv")
    test_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    num_batches = len(indices)//batch_size
    last_batch_len = len(indices)%batch_size

    model.eval()
    model.to(device)

    print(f" [INFO] Model: {model_name} loaded.")

    with torch.no_grad():
        print(f" [INFO] Getting top5 probabilities for {model_name} on {dataset} dataset")
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            a = i*batch_size
            b = (i+1)*batch_size
            if i == num_batches:
                b = a + last_batch_len
                batch_size = last_batch_len

            for j in indices[a:b]:
                top5prob = torch.nn.functional.softmax(output, dim=1)
                top5prob = top5prob.cpu()
                top5prob = torch.topk(top5prob, k=5)
                for k in range(batch_size):
                    path = f"data/top5prob/{model_name}_{dataset}_{precision}_top5prob_{a+k}.pt"
                    tensor = torch.cat((top5prob.indices[k].unsqueeze(0), top5prob.values[k].unsqueeze(0)), dim=0)
                    torch.save(tensor, path)
        print(f" [INFO] Top5 probabilities saved in data/top5prob/{model_name}_{dataset}_{precision}_top5prob_*.pt")


if __name__ == '__main__':
    main()