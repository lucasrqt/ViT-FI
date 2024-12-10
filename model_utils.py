import timm
import torch
from torchvision import datasets as tv_datasets
from torchvision import transforms as tv_transforms
import configs
from typing import Dict
from torch.utils.data import Subset
import pandas as pd

def get_vit_config(model) -> Dict:
    return timm.data.resolve_data_config({}, model=model)

def get_vit_transforms(model, precision) -> tv_transforms.Compose:
    transforms = timm.data.transforms_factory.create_transform(**get_vit_config(model))

    if precision == configs.FP16:
        class CustomToFP16:
            def __call__(self, tensor_in):
                return tensor_in.type(torch.float16)

        transforms = transforms.transforms.insert(-1, CustomToFP16())

    return transforms

def get_model(model_name: str, precision) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=True)
    if precision == configs.FP16:
        model = model.half()

    model.eval()
    model.zero_grad(set_to_none=True)
    return model

def get_dataset(dataset_name: str, transforms: tv_transforms.Compose, batch_size: int) -> torch.utils.data.DataLoader:
    dataset_path = configs.DATASETS_DIRS[dataset_name]
    if dataset_path is None:
        raise ValueError(f"Dataset path for {dataset_name} is not defined.")
    
    test_set = tv_datasets.imagenet.ImageNet(root=dataset_path, transform=transforms,
                                                 split='val')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_set, test_loader

def get_correct_indices(test_set, file) -> Subset:
    df = pd.read_csv(file, index_col=0)
    indices = df.index.tolist()
    return indices, Subset(test_set, indices)