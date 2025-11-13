#! /usr/bin/env bash

script="main.py"

# Run the tests
models=(
    "vit_base_patch16_224"
    "swin_base_patch4_window7_224"
    "gpt2"
    "facebook/bart-large-mnli"
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

gpt2_microops=(
    "GPT2Block"
    "GPT2Attention"
    "GPT2MLP"
)

bart_microops=(
    "BartEncoderLayer"
    "BartDecoderLayer"
    "BartSdpaAttention"
    "BartMlp"
)

injection_types=(
    # "RANDOM"
    # "FIXED"
    # "SINGLE"
    "ROW"
    "COL"
    # "SINGLE_RANDOM"
)

bitflip_positions=(
    "DUMMY"  # to be replaced in the script
    # "14"
	# "15"
	# "16"
	# "17"
	# "18"
	# "19"
	# "20"
	# "21"
	# "22"
	# "23"
)

device="cuda:0"
# dataset="imagenet"
# dataset="glue_mnli"
batchsize=32
# seed=0
seeds=(
    # 0
    493
    # 666
    # 31417
    # 182036
    # 29052001
    # 35014520
    # 4294967295
    # 2796017452
    # 1084398730
    #---
    # 3208799631
    # 2357136044
    # 2546248239
    # 3071714933
    # 3626093760
    # 2588848963
    # 3684848379
    # 2340255427
    # 3638918503
    # 1819583497
    # 2678185683
    # 2774094101
    # 1650906866
    # 1879422756
    # 1277901399
    #---
    # 3830135878
    # 243580376
    # 4138900056
    # 1171049868
    # 1646868794
    # 2051556033
    # 3400433126
    # 3488238119
    # 2271586391
    # 2061486254
    # 2439732824
    # 1686997841
    # 3975407269
    # 3590930969
    # 305097549
    # 1449105480
    # 374217481
    # 2783877012
    # 86837363
    # 1581585360
    # 3576074995
    # 4110950085
    # 3342157822
    # 602801999
    # 3736673711
    # half of seeds
    # 3736996288
    # 4203133778
    # 2034131043
    # 3432359896
    # 3439885489
    # 1982038771
    # 2235433757
    # 3352347283
    # 2915765395
    # 507984782
    # 3095093671
    # 2748439840
    # 2499755969
    # 615697673
    # 2308000441
    # 4057322111
    # 3258229280
    # 2241321503
    # 454869706
    # 1780959476
    # 2034098327
    # 1136257699
    # 800291326
    # 3325308363
    # 3165039474
    # 1959150775
    # 930076700
    # 2441405218
    # 580757632
    # 80701568
    # 1392175012
    # 2652724277
    # 642848645
    # 2628931110
    # 954863080
    # 2649711348
    # 1659957521
    # 4053367119
    # 3876630916
    # 2928395881
    # 1932520490
    # 1544074682
    # 2633087519
    # 1877037944
    # 3875557633
    # 2996303169
    # 426405863
    # 258666409
    # 4165298233
    # 2863741219
)

default_targets=( # for vit_base_patch16_224 and gpt2 (12 blocks)
    # "FIRST"
    # "LAST"
    # "MIDDLE"
    # "MIDDLE_HALF"
    # "BEFORE_LAST"
    "0" # first layer
    "5" # middle layer
    "11" # last layer
)

# bart_attentions=34
# bart_encoders=12
# bart_decoders=12
# bart_mlps=24

# swin_total_layers=24
# vit_total_layers=12
# gpt2_total_layers=12

range_restriction_modes=(
    # "NONE"
    # "CLAMP"
    "TO_ZERO"
)

# options="--inject-on-correct-predictions --load-critical --save-critical-logits"
# options="--inject-on-correct-predictions --shuffle-dataset"
options="--verbose --inject-on-correct-predictions"
# options="--nsamples 2048 --verbose --inject-on-correct-predictions"
# options="--nsamples 32 --verbose --inject-on-correct-predictions"

# creating folder for results
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
mkdir -p data/"$current_time"_campaign

#### python pattern for file_names :
## base_name = f"{model_name}-{dataset_name}-{precision}-{microop}-{float_threshold_FM}-{seed}-layer_{str(layer)}-it_{str(it)}-rrmode_{str(rr_mode)}"
## if seed_specific is not None:
##     base_name += f"-seedspecif_{seed_specific}"
## return f"{base_name}.csv"

for model in "${models[@]}"; do
    if [[ $model == "gpt2" || $model == "facebook/bart-large-mnli" ]]; then
        dataset="glue_mnli"
    else
        dataset="imagenet"
    fi
    for prec in "${precision[@]}"; do
        for threshold in "${float_thresholds[@]}"; do
            for seed in "${seeds[@]}"; do
                for it in "${injection_types[@]}"; do
                    for rrm in "${range_restriction_modes[@]}"; do
                        options+=" --range-restriction-mode $rrm"
                        for bitflip in "${bitflip_positions[@]}"; do
                            it_for_filename=$it
                            if [[ $it == "SINGLE" ]]; then
                                options+=" --bitflip-position $bitflip"
                                it_for_filename+="--bit$bitflip"
                            fi
                            if [[ $model == "swin"* ]]; then
                                for microop in "${swin_microops[@]}"; do
                                    targets=(
                                        "0" # first layer
                                        "11" # middle layer
                                        "23" # last layer
                                    )
                                    for target in "${targets[@]}"; do
                                        time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                        # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                        mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv data/"$current_time"_campaign/
                                    done
                                done
                            elif [[ $model == "gpt2" ]]; then
                                for microop in "${gpt2_microops[@]}"; do
                                    for target in "${default_targets[@]}"; do
                                        time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                        # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                        mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv data/"$current_time"_campaign/
                                    done
                                done
                            elif [[ $model == "facebook/bart-large-mnli" ]]; then
                                for microop in "${bart_microops[@]}"; do
                                    if [[ $microop == "BartEncoderLayer" || $microop == "BartDecoderLayer" ]]; then
                                        targets=$default_targets
                                    elif [[ $microop == "BartSdpaAttention" ]]; then
                                        targets=(
                                            "0" # first layer
                                            "17" # middle layer
                                            "33" # last layer
                                        )
                                    elif [[ $microop == "BartMlp" ]]; then
                                        targets=(
                                            "0" # first layer
                                            "11" # middle layer
                                            "23" # last layer
                                        )
                                    fi
                                    for target in "${targets[@]}"; do
                                        time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                        # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                        model_name=${model//\//_}
                                        mv data/"$model_name"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv data/"$current_time"_campaign/
                                    done
                                done
                            else
                                for microop in "${vit_microops[@]}"; do
                                    for target in "${default_targets[@]}"; do
                                        time python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop --injection-type $it $options
                                        # echo "python3 $script --model $model --precision $prec --fault-model-threshold $threshold --device $device --dataset $dataset --batch-size $batchsize --seed $seed --target-layer $target --microop $microop $options"
                                        mv data/"$model"-"$dataset"-"$prec"-"$microop"-*-"$seed"-layer_"$target"-it_"$it_for_filename"-rrmode_"$rrm".csv data/"$current_time"_campaign/
                                    done
                                done
                            fi
                        done
                    done
                done
            done
        done
    done
done
