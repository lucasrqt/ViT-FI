#! /usr/bin/env bash

script="grounding_dino_tests.py"

# Run the tests
models=(
    "groundingdino_swint_ogc"
    # "groundingdino_swinb_cogcoor"
)

iou_thresholds=(
    # "0.5"
    "0.75"
)

device="cuda:0"
batchsize=1
dataset="coco"


for model in "${models[@]}"; do
    for th in "${iou_thresholds[@]}"; do
        echo "Model: $model, IoU Threshold: $th"
        python $script --model $model --iou-threshold $th -d $device -D $dataset --batch-size $batchsize
    done
done
