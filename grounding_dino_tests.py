#! /usr/bin/env python3

import configs

from compare_utils import calculate_iou
from compare_utils import count_elements

import os
import sys
import time

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

def main() -> None:
    args = parse_args()

    model_name = args.model
    batch_size = args.batch_size
    device = args.device

    model_config_path = configs.VITS_MULTIMODAL_CONFIGS_PATHS[model_name]

    cfg, model = load_model(model_config_path, f"{configs.GROUNDING_DINO_WEIGHTS_PATH}{model_name}.pth", device)
    model = model.to(device)
    dataset, dataloader, caption = load_dataset(batch_size)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    for i, (images, targets) in enumerate(dataloader):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        
        for image_id in cocogrounding_res.keys():
            print(f"Image ID: {image_id}")
            scores = cocogrounding_res[image_id]["scores"]
            boxes = cocogrounding_res[image_id]["boxes"]
            labels = cocogrounding_res[image_id]["labels"]
            print(f"Scores: {scores.shape}, Boxes: {boxes.shape}, Labels: {labels.shape}")
            print("\n")
        evaluator.update(cocogrounding_res)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(dataloader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(dataloader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

        if i == 10:
            break

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

if __name__ == "__main__":
    main()