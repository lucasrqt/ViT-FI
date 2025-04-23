#! /usr/bin/env bash

script="main.py"

# Run the tests
models=(
   "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
)

precision=(
    "fp32"
    # "fp16",
)

float_thresholds=(
    "1e-04"
    # "1e-03"
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

injection_types=(
    "RANDOM"
    "FIXED"
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
    182036
    # 29052001
    35014520
)

targets=(
    # "FIRST"
    "LAST"
    # "MIDDLE"
)

# options="--inject-on-correct-predictions --load-critical --save-critical-logits"
# options="--inject-on-correct-predictions --shuffle-dataset"
options="--inject-on-correct-predictions"

# creating folder for results
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
mkdir -p data/"$current_time"_campaign

for model in "${models[@]}"; do
    for prec in "${precision[@]}"; do
        for threshold in "${float_thresholds[@]}"; do
            for seed in "${seeds[@]}"; do
                for target in "${targets[@]}"; do
                    for it in "${injection_types[@]}"; do
                        if [[ $model == "swin"* ]]; then
                            for microop in "${swin_microops[@]}"; do
                                time python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                # echo "python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                mv data/"$model"_"$dataset"_"$prec"_"$microop"_*_"$seed"_layer-"$target"_it-"$it".csv data/"$current_time"_campaign/
                            done
                        else
                            for microop in "${vit_microops[@]}"; do
                                time python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                # echo "python $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                mv data/"$model"_"$dataset"_"$prec"_"$microop"_*_"$seed"_layer-"$target"_it-"$it".csv data/"$current_time"_campaign/
                            done
                        fi
                    done
                done
            done
        done
    done
done
