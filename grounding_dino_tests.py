#! /usr/bin/env python3

import configs

from compare_utils import calculate_iou
from compare_utils import count_elements

import os
import sys

sys.path.extend([
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "GroundingDINO"),
])

import GroundingDINO.groundingdino.datasets.transforms as gdino_transforms
from GroundingDINO.demo.test_ap_on_coco import CocoDetection as GDINOCocoDetection
from GroundingDINO.demo.test_ap_on_coco import PostProcessCocoGrounding as GDINOPostProcessCocoGrounding
from GroundingDINO.groundingdino.datasets.cocogrounding_eval import (
    CocoGroundingEvaluator as GDINOCocoGroundingEvaluator
)
from GroundingDINO.groundingdino.models import build_model as gdino_build_model
from GroundingDINO.groundingdino.util import get_tokenlizer as gdino_get_tokenlizer
from GroundingDINO.groundingdino.util.misc import collate_fn as gdino_collate_fn
from GroundingDINO.groundingdino.util.slconfig import SLConfig as gdino_SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict as gdino_clean_state_dict
from GroundingDINO.groundingdino.util.utils import get_phrases_from_posmap as gdino_get_phrases_from_posmap

# For custom datasets
from GroundingDINO.demo.inference_on_a_image import load_image as gdino_load_image
from GroundingDINO.demo.inference_on_a_image import plot_boxes_to_image as gdino_plot_boxes_to_image

import torch

def load_dataset(batch_size) -> None:
    # COCO default transformations
    transform = gdino_transforms.Compose(
        [
            gdino_transforms.RandomResize([800], max_size=1333),
            gdino_transforms.ToTensor(),
            gdino_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = GDINOCocoDetection(configs.COCO_DATASET_VAL, configs.COCO_DATASET_ANNOTATIONS,
                                    transforms=transform)
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1,
                                                collate_fn=gdino_collate_fn)

    # build captions
    # caption = self.input_captions
    # if self.input_captions == '':
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = gdino_SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = gdino_build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(gdino_clean_state_dict(checkpoint["model"]), strict=False)
    # print(load_res)
    _ = model.eval()
    return model

def main() -> None:
    pass

if __name__ == "__main__":
    main()