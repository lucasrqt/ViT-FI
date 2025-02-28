#! /usr/bin/env bash

script="main.py"

# Run the tests
models=(
    "vit_base_patch16_224"
#    "swin_base_patch4_window7_224"
)

precision=(
    "fp32"
    # "fp16",
)

float_thresholds=(
    # "1e-04"
    "1e-03"
)

swin_microops=(
    "SwinTransformerBlock"
    "Mlp"
    "WindowAttention"
)

vit_microops=(
    "Block"
    "Attention"
    "Mlp"
)

device="cuda:0"
dataset="imagenet"
batchsize=32
# seed=0
seeds=(
    # 0
    # 493
    # 666
    # 31417
    # 182036
    # 29052001
    # 35014520
    4294967295
    2796017452
    1084398730
    3208799631
)

targets=(
    "FIRST"
    "LAST"
    "MIDDLE"
)

# options="--inject-on-correct-predictions --load-critical --save-critical-logits"
options="--inject-on-correct-predictions"

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for threshold in "${float_thresholds[@]}"; do
            for seed in "${seeds[@]}"; do
                for target in "${targets[@]}"; do
                    if [[ $model == "swin"* ]]; then
                        for microop in "${swin_microops[@]}"; do
                            time python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options
                        done
                    else
                        for microop in "${vit_microops[@]}"; do
                            time python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options
                        done
                    fi
                done
            done
        done
    done
done
