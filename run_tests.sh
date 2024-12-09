#! /usr/bin/env bash

script="test.py"

# Run the tests
models=(
    "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
)

precision=(
    "fp32"
    # "fp16"
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


for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for threshold in "${float_thresholds[@]}"; do
            if [ "$model" == "swin_base_patch4_window7_224" ]; then
                for microop in "${swin_microops[@]}"; do
                    echo "Model: $model, Precision: $prec, Threshold: $threshold, Microop: $microop"
                    python $script -m $model -p $prec --fault-model-threshold $threshold -M $microop -d $device -D $dataset -b $batchsize
                done
            else
                for microop in "${vit_microops[@]}"; do
                    echo "Model: $model, Precision: $prec, Threshold: $threshold, Microop: $microop"
                    python $script -m $model -p $prec --fault-model-threshold $threshold -M $microop -d $device -D $dataset -b $batchsize
                done
            fi
        done
    done
done