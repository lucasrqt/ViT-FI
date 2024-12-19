#! /usr/bin/env python3

import configs

from compare_utils import calculate_iou
from compare_utils import count_elements

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.extend([
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO"),
])

import GroundingDINO.groundingdino.datasets.transforms as gdino_transforms
from GroundingDINO.demo.test_ap_on_coco import CocoDetection
from GroundingDINO.demo.test_ap_on_coco import PostProcessCocoGrounding
from GroundingDINO.groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

from GroundingDINO.groundingdino.models import build_model as gdino_build_model
from GroundingDINO.groundingdino.util import get_tokenlizer
from GroundingDINO.groundingdino.util.misc import collate_fn as gdino_collate_fn
from GroundingDINO.groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap 
from groundingdino.util import box_ops
from typing import Tuple

# For custom datasets
# from GroundingDINO.demo.inference_on_a_image import load_image
# from GroundingDINO.demo.inference_on_a_image import plot_boxes_to_image 
# from GroundingDINO.groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span


import torch
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grounding DINO tests", add_help=True)
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("-m", "--model", type=str, default=configs.GROUNDING_DINO_SWINT_OGC, help="Model name.", choices=configs.VITS_MULTIMODAL_CONFIGS)
    parser.add_argument("-d", "--device", type=str, default=configs.GPU_DEVICE, help="Device to run the model.", choices=[configs.CPU, configs.GPU_DEVICE])
    parser.add_argument("-D", "--dataset", type=str, default=configs.COCO, help="Dataset name.", choices=[configs.COCO])
    parser.add_argument("--box-threshold", type=float, default=configs.BOX_THRESHOLD, help="Box threshold.")
    parser.add_argument("--text-threshold", type=float, default=configs.TEXT_THRESHOLD, help="Text threshold.")
    parser.add_argument("--iou-threshold", type=float, default=configs.IOU_THRESHOLD, help="IOU threshold.")
    parser.add_argument("--subset-path", type=str, default=None, help="Subset path.")
    parser.add_argument("--confidence", type=str, default="low", help="Confidence on inputs.", choices=["low", "high"])
    return parser.parse_args()

def load_dataset(batch_size) -> None:
    # COCO default transformations
    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS,
                                    transforms=transform)
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                                collate_fn=gdino_collate_fn)
    
    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    return dataset, test_loader, caption
    
    

def load_model(model_config_path, model_checkpoint_path, device):
    args = gdino_SLConfig.fromfile(model_config_path)
    args.device = device
    model = gdino_build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return args, model

# def get_ground_dino_outputs(outputs: torch.Tensor, tokenlizer, caption: str, box_threshold: float, text_threshold: float) -> Tuple[torch.Tensor, list]:
#     caption = caption.lower()
#     caption = caption.strip()
#     if not caption.endswith("."):
#         caption = caption + "."
#     logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
#     boxes = outputs["pred_boxes"][0]  # (nq, 4)

#     # filter output
#     # if token_spans is None:
#     logits_filt = logits.cpu().clone()
#     boxes_filt = boxes.cpu().clone()
#     filt_mask = logits_filt.max(dim=1)[0] > box_threshold
#     logits_filt = logits_filt[filt_mask]  # num_filt, 256
#     boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

#     # get phrase
#     # tokenlizer = model.tokenizer
#     tokenized = tokenlizer(caption)
#     # build pred
#     pred_phrases = []
#     for logit, box in zip(logits_filt, boxes_filt):
#         pred_phrase = get_phrases_from_posmap(
#             logit > text_threshold, tokenized, tokenlizer)
#         pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

#     return boxes_filt, pred_phrases

def normalize_boxes(boxes, target_sizes):
    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]
    return boxes

def compute_f1_per_image(eval_imgs, iou_threshold):
    f1_scores = {}

    for img_idx, eval_img in enumerate(eval_imgs["bbox"]):
        if eval_img is None:
            continue  # Skip if no evaluation data for this image
        
        # Process each evaluation entry (category-specific)
        for data in eval_img:
            if data is None:
                continue  # Skip if no evaluation data for this category in the image
            
            # Extract data
            data = data[0]
            gt_matches = data["gtMatches"][0]  # Matches to ground truth
            print(f'{gt_matches = }')
            dt_matches = data["dtMatches"][0]  # Matches to detections
            dt_scores = np.array(data["dtScores"])
            gt_ignore = data["gtIgnore"]
            dt_ignore = data["dtIgnore"][0]

            # Apply IoU threshold: Keep matches above the threshold
            valid_matches = dt_matches > 0  # Detections that matched any ground truth
            dt_iou_matches = (dt_matches >= iou_threshold) & valid_matches

            # True Positives (TP): Valid detections that meet the IoU threshold
            TP = np.sum(dt_iou_matches & (~dt_ignore))
            
            # False Positives (FP): Detections not matching any ground truth or below IoU threshold
            FP = np.sum((~dt_iou_matches) & (~dt_ignore))
            
            # False Negatives (FN): Ground truths not matched by any detection
            FN = np.sum((gt_matches < iou_threshold) & (~gt_ignore))
            
            # Compute Precision and Recall
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            # Compute F1 Score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store results
            f1_scores[data["image_id"]] = {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            }
    
    return f1_scores

@torch.no_grad()
def main() -> None:
    args = parse_args()

    model_name = args.model
    batch_size = args.batch_size
    device = args.device
    iou_threshold = args.iou_threshold
    subset_path = args.subset_path
    confidence = args.confidence

    model_config_path = configs.VITS_MULTIMODAL_CONFIGS_PATHS[model_name]

    cfg, model = load_model(model_config_path, f"{configs.GROUNDING_DINO_WEIGHTS_PATH}{model_name}.pth", device)
    model = model.to(device)
    dataset, dataloader, caption = load_dataset(batch_size)
    if subset_path:
        df = pd.read_csv(subset_path)
        if confidence == "low":
            indices = df.index.tolist()[-3:]
        else:
            indices = df.index.tolist()[:3]
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=1,
                                                    collate_fn=gdino_collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)
    
    # evaluator.coco_eval["bbox"].params.iouThrs = np.array([iou_threshold])

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    print(len(dataloader))
    pred_boxes_lens = []

    eval_results = []
    for i, (images, targets) in enumerate(dataloader):
        evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)
    
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        outputs = model(images, captions=input_captions)

        # print("="*50)
        # print(f"Batch {i+1}/{len(dataloader)}")
        # boxes, labels = get_ground_dino_outputs(outputs, tokenlizer, caption, args.box_threshold, args.text_threshold)
        # print(f'{boxes = }, {boxes.shape = }')
        # print(f'{labels = }')

        # break

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)

        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}

        for j in range(len(targets)):
            pred_boxes_lens.append(len(targets[j]["boxes"]))
        
        # Visualize the boxes on the image
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches

        # for idx in range(bs):
        #     fig, ax = plt.subplots(1)
        #     image = images[idx].cpu().numpy().transpose(1, 2, 0)
        #     image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        #     ax.imshow(image)

        #     boxes = results[idx]["boxes"].cpu().numpy()
        #     for box in boxes:
        #         x_min, y_min, x_max, y_max = box
        #         rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        #         ax.add_patch(rect)

        #     plt.savefig(f"data/{model_name}_{iou_threshold}_batch{i}_img{idx}.png")

        evaluator.update(cocogrounding_res)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(dataloader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i+1}/{len(dataloader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")


        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        stats = evaluator.coco_eval["bbox"].stats.tolist()
        eval_results.append({
            "image_idx": i,
            "AP@IoU[0.5:0.95]": stats[0],
            "AP@IoU[0.5]": stats[1],
            "AP@IoU[0.75]": stats[2],
            "AP@IoU[0.5:0.95]_small": stats[3],
            "AP@IoU[0.5:0.95]_medium": stats[4],
            "AP@IoU[0.5:0.95]_large": stats[5],
            "AR@IoU[0.5:0.95]": stats[6],
            "AR@IoU[0.5]": stats[7],
            "AR@IoU[0.75]": stats[8],
            "AR@IoU[0.5:0.95]_small": stats[9],
            "AR@IoU[0.5:0.95]_medium": stats[10],
            "AR@IoU[0.5:0.95]_large": stats[11],
        })

        if i == 2499:
            break

    # Save results to a CSV
    results_df = pd.DataFrame(eval_results)
    results_df.to_csv(f"data/{model_name}_{batch_size}.csv", index=False)
    
    # stop = time.time()

    # print(f"Total time: {stop - start:.2f}s")


    # all_results = []  # Store results for CSV

    # category_mapping = {item["id"]: item["name"] for item in dataset.coco.dataset['categories']}

    # # print(len(evaluator.eval_imgs["bbox"]))
    # # for i in range(0, len(evaluator.eval_imgs["bbox"])):
    # #     # correct_boxes_len = pred_boxes_lens[i]  # Length of predicted boxes in this batch
    # #     for j in range(len(evaluator.eval_imgs["bbox"][i])):
    # #         data = evaluator.eval_imgs["bbox"][i][j][0, 0]

    # #         if data is None:
    # #             continue

    # #         image_id = data["image_id"]
    # #         category_id = data["category_id"]
    # #         dt_ids = data["dtIds"]
    # #         dt_scores = data["dtScores"]
    # #         dt_matches = data["dtMatches"][0]  # Extract matching array for detections
            
    # #         # Loop through each detection
    # #         for dt_id, score, match in zip(dt_ids, dt_scores, dt_matches):
    # #             all_results.append({
    # #                 "index": i,
    # #                 "image_id": image_id,
    # #                 "category_id": category_id,
    # #                 "category_name": category_mapping.get(category_id, "unknown"),
    # #                 "dt_id": dt_id,
    # #                 "score": score,
    # #                 "match": match,
    # #             })

    # Save results to a CSV
    # results_df = pd.DataFrame(all_results)
    # results_df.to_csv(f"data/{model_name}_{batch_size}_{iou_threshold}.csv", index=False)
    # print("Results saved to detection_results.csv")

    # print(evaluator.coco_eval["bbox"].eval["precision"])

    # evaluator.synchronize_between_processes()
    # evaluator.accumulate()
    # evaluator.summarize()

    # f1_results = compute_f1_per_image(evaluator.eval_imgs, iou_threshold)
    # for image_id, metrics in f1_results.items():
    #     print(f"Image ID: {image_id}")
    #     print(metrics)

    # print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

if __name__ == "__main__":
    main()