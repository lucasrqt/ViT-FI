#! /usr/bin/env bash

script="select_inputs.py"

# Run the tests
models=(
   "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
)

precision=(
    "fp32"
    # "fp16",
)

input_selection_methods=(
    "VARIANCE"
    # "MAX_P"
    # "CONFIDENCE"
    # "DSA"
)

device="cuda:0"
dataset="imagenet"
batchsize=32
# seed=0
seeds=(
    0
    # 493
    # 666
    # 31417
    # 182036
    # 29052001
    # 35014520
)

options="--load-correct-predictions"

# creating folder for results
cd input_selection

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for method in "${input_selection_methods[@]}"; do
            for seed in "${seeds[@]}"; do
                time python $script --model $model --precision $prec --method $method --device $device --dataset $dataset --batch-size $batchsize --seed $seed $options
            done
        done
    done
done
