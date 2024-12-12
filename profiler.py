#! /usr/bin/env python3

import configs
import model_utils
from torch.profiler import profile, record_function, ProfilerActivity

MODELS = [
    configs.VIT_BASE_PATCH16_224,
    configs.SWIN_BASE_PATCH4_WINDOW7_224,
]

BATCH_SIZE = 32
DATASET = "imagenet"

def main() -> None:
    sort_by_keyword = configs.GPU_DEVICE[:-2] + "_time_total"
    activities = [ProfilerActivity.CUDA]
    for model_name in MODELS:
        model = model_utils.get_model(model_name, configs.FP32)
        transforms = model_utils.get_vit_transforms(model, configs.FP32)
        _, test_loader = model_utils.get_dataset(DATASET, transforms, BATCH_SIZE)
        model = model.to(configs.GPU_DEVICE)

        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(configs.GPU_DEVICE)

        with profile(activities=activities, record_shapes=True) as prof:
            with record_function("model_inference"):
                model(inputs)
        
        print("="*20, model_name, "="*20)
        print(prof.key_averages().table(sort_by=sort_by_keyword))
        prof.export_chrome_trace(f"data/profiles/{model_name}_trace.json")



if __name__ == "__main__":
    main()