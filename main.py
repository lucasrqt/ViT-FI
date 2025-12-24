#! /usr/bin/env python3

import configs
import statistical_fi
import utils.model_utils as model_utils
import utils.result_data_utils as result_data_utils
import torch
import pandas as pd
import numpy as np
import time
from statistics import mean
import os
import cli.logger_formatter as logger_formatter
from cli.parsers import MainParser

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


def run_injections(
    model_name,
    dataset_name,
    microop,
    model,
    model_for_fault,
    data_loader,
    precision,
    device,
    batch_size,
    result_df,
    result_file,
    logger,
) -> None:
    global TIME_MEASURE
    model.eval()
    model.to(device)

    model_for_fault.eval()
    model_for_fault.to(device)

    if model_for_fault is None:
        raise ValueError("Model for fault injection is not defined.")

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()

        images = images.to(device)
        labels = labels.to(device)

        # microop = statistical_fi.select_microop(model_name)
        out_wo_fault, out_prob_wo_fault = statistical_fi.run_inference(
            model, images, device
        )
        out_wo_fault, out_prob_wo_fault = (
            out_wo_fault.squeeze(),
            out_prob_wo_fault.squeeze(),
        )
        out_with_fault, out_prob_w_fault = statistical_fi.run_inference(
            model_for_fault, images, device
        )
        out_with_fault, out_prob_w_fault = (
            out_with_fault.squeeze(),
            out_prob_w_fault.squeeze(),
        )
        # return

        # print(out_prob_w_fault, out_with_fault)

        logger.warning("-" * 80)
        logger.warning(f"Batch {i} - Microop: {microop}")
        for j in range(len(images)):
            if out_wo_fault[j].item() != out_with_fault[j].item():
                logger.warning(
                    f"CRITICAL {(i*batch_size)+j+1} - Ground truth: {labels[j]} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()} (confidence: {(out_prob_wo_fault[j][0].item() - out_prob_wo_fault[j][1].item()):.4f})"
                )
            # logger.debug(f"(confidence: {(out_prob_wo_fault[j][0].item() - out_prob_wo_fault[j][1].item()):.4f})")

            result_df = result_data_utils.append_row(
                result_df,
                labels[j].item(),
                out_wo_fault[j].item(),
                out_with_fault[j].item(),
                out_prob_wo_fault[j][0].item(),
                out_prob_wo_fault[j][1].item(),
                out_prob_w_fault[j][0].item(),
                out_prob_w_fault[j][1].item(),
            )
            result_data_utils.save_result_data(
                pd.DataFrame(result_df), configs.RESULTS_DIR, result_file
            )

        # if i == 9:
        #     logger.info(f"Stopping after {i+1} batches.")
        #     break

    logger.info("Done.")

def run_injections_gpt2(
    model_name,
    dataset_name,
    microop,
    model,
    model_for_fault,
    data_loader,
    precision,
    device,
    batch_size,
    result_df,
    result_file,
    logger,
) -> None:
    model.eval()
    model.to(device)

    model_for_fault.eval()
    model_for_fault.to(device)

    if model_for_fault is None:
        raise ValueError("Model for fault injection is not defined.")

    for i, batch in enumerate(data_loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].cpu().numpy()

        # microop = statistical_fi.select_microop(model_name)
        out_wo_fault, out_prob_wo_fault = statistical_fi.run_inference_gpt2(
            model, inputs, device
        )
        out_wo_fault, out_prob_wo_fault = (
            out_wo_fault.squeeze(),
            out_prob_wo_fault.squeeze(),
        )
        out_with_fault, out_prob_w_fault = statistical_fi.run_inference_gpt2(
            model_for_fault, inputs, device
        )
        out_with_fault, out_prob_w_fault = (
            out_with_fault.squeeze(),
            out_prob_w_fault.squeeze(),
        )
        # return

        # print(out_prob_w_fault, out_with_fault)

        logger.warning("-" * 80)
        logger.warning(f"Batch {i} - Microop: {microop}")
        for j in range(len(labels)):
            if out_wo_fault[j].item() != out_with_fault[j].item():
                logger.warning(
                    f"CRITICAL {(i*batch_size)+j+1} - Ground truth: {labels[j]} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()} (confidence: {(out_prob_wo_fault[j][0].item() - out_prob_wo_fault[j][1].item()):.4f})"
                )

            result_df = result_data_utils.append_row(
                result_df,
                labels[j].item(),
                out_wo_fault[j].item(),
                out_with_fault[j].item(),
                out_prob_wo_fault[j][0].item(),
                out_prob_wo_fault[j][1].item(),
                out_prob_w_fault[j][0].item(),
                out_prob_w_fault[j][1].item(),
            )
            result_data_utils.save_result_data(
                pd.DataFrame(result_df), configs.RESULTS_DIR, result_file
            )
        # if i == 9:
        #     logger.info(f"Stopping after {i+1} batches.")
        #     break

    logger.info("Done.")

def get_faulty_top5(
    model_name, microop, model, data_loader, precision, device, batch_size, logger
) -> None:
    model.eval()
    model.to(device)

    start = time.time()
    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()

        images = images.to(device)
        labels = labels.to(device)

        # microop = statistical_fi.select_microop(model_name)
        with torch.no_grad():
            out_with_fault = model(images)
            labels = labels
            if "cuda" in device:
                torch.cuda.synchronize()

            logger.debug("-" * 80)
            logger.debug(f"Batch {i} - Microop: {microop}")

            top5prob = torch.nn.functional.softmax(out_with_fault, dim=1)
            top5prob = top5prob.cpu()
            top5prob = torch.topk(top5prob, k=5)
            for j in range(len(images)):
                path = f"data/top5prob/faulty-{model_name}-{microop}-top5prob_{(i*batch_size)+j}.pt"
                tensor = torch.cat(
                    (top5prob.indices[j].unsqueeze(0), top5prob.values[j].unsqueeze(0)),
                    dim=0,
                )
                torch.save(tensor, path)
                logger.debug(f"Image {(i*batch_size)+j+1} saved.")

    end = time.time()
    logger.debug(f"Time for full pass: {end-start}s")

    logger.info("Done.")


def main() -> None:
    parser = MainParser()
    args = parser.parse_args()

    # Parse arguments
    precision = args.precision
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    shuffle_dataset = args.shuffle_dataset
    model_name = args.model
    dataset_name = args.dataset
    fault_model_threshold = f"{args.fault_model_threshold:.2e}"
    microop = args.microop
    inject_on_corr_preds = args.inject_on_correct_predictions
    save_critical_logits = args.save_critical_logits
    save_top5prob = args.save_top5prob
    target_layer = args.target_layer
    verbose = args.verbose
    injection_type = args.injection_type
    specific_seed = args.specific_seed
    nsamples = args.nsamples
    bitflip_position = args.bitflip_position
    range_restriction_mode = args.range_restriction_mode

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if microop is None:
        raise ValueError("Microoperation not defined.")

    if statistical_fi.check_microop(model_name, microop) is False:
        raise ValueError(
            f"Microoperation {microop} not supported by the model {model_name}."
        )


    firstlog_msg = f"Model {model_name}, microop {microop} on dataset {dataset_name} selected with {str(injection_type)}"
    if injection_type == statistical_fi.InjectionType.SINGLE:
        firstlog_msg += f" (bitflip on bit {bitflip_position})"
    firstlog_msg += "."
    logger.info(firstlog_msg)
    if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        model = model_utils.get_model(model_name, precision)
        model_for_fault = model_utils.get_model(model_name, precision)
        transforms = model_utils.get_vit_transforms(model, precision)

        #### TEST CASE
        # dummy_input = torch.randn(32, 3, 224, 224)
        # out_wo_fault = statistical_fi.run_inference(model, dummy_input, device).squeeze()
        # out_with_fault = statistical_fi.run_inference(model_for_fault, dummy_input, device).squeeze()

        # print("-" * 80)
        # print(f" [+] Batch {0} - Microop: {microop}")
        # for j in range(len(dummy_input)):
        #     print(f" [+] Image {j+1} - Ground truth: {out_wo_fault[j].item()} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}")
        ####

        test_set, data_loader = model_utils.get_dataset(
            dataset_name, transforms, batch_size, shuffle=shuffle_dataset
        )
        if inject_on_corr_preds:
            # _, subset = model_utils.get_correct_indices(
            #     test_set,
            #     f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv",
            # )

            # inject on low_confidence correct predictions
            df = pd.read_csv(f"data/{model_name}-{dataset_name}-predictions.csv", index_col=0)
            df = df.sort_values(by="confidence", ascending=True)
            if nsamples > 0:
                df = df.head(nsamples)
            indices = df.index.tolist()
            subset = torch.utils.data.Subset(test_set, indices)

            data_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle_dataset,
            )

            if args.load_critical:
                df = pd.read_csv("data/fi_critical_images.csv")
                df = df[(df["model"] == model_name) & (df["microop"] == microop)]
                if df.empty:
                    raise ValueError("No critical images found.")
                indices = df["image_id"].tolist()
                # full_batchs = []
                batch_indices = []
                for index in indices:
                    batch_id = model_utils.get_batch_id(index, batch_size)
                    batch_indices.append(batch_id)
                #     full_batchs += range(batch_id*batch_size, (batch_id+1)*batch_size)
                # subset = Subset(subset, full_batchs)

            logger.info(f"{len(subset)} correct predictions found.")
            # data_loader = torch.utils.data.DataLoader(
            #     subset, batch_size=batch_size, shuffle=shuffle_dataset
            # )
            logger.info("Injecting faults on correct predictions only.")
        # if specific_seed is not None:
        #     sampler_generator = torch.Generator(device=configs.CPU)
        #     sampler_generator.manual_seed(specific_seed)

        #     subset = torch.utils.data.RandomSampler(data_source=test_set, replacement=False, num_samples=batch_size,
        #                                             generator=sampler_generator)
        #     data_loader = torch.utils.data.DataLoader(dataset=test_set, sampler=subset, batch_size=batch_size,
        #                                             shuffle=False, pin_memory=True)
    elif model_name == configs.FACEBOOK_BART:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model_for_fault = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        model_for_fault.eval()

        model_name = model_name.replace("/", "_")  # For saving files

        # Load dataset (MNLI)
        if dataset_name not in [configs.GLUE_MNLI]:
            raise ValueError(f"{model_name} only supports MNLI dataset for now.")
        
        dataset, task = dataset_name.split("_")
        raw_dataset = load_dataset(dataset, task)
        raw_dataset = raw_dataset.filter(lambda x: x["label"] != -1)

        # Preprocess MNLI samples (premise, hypothesis)
        # ----------------------------
        # Task-specific preprocessing
        # ----------------------------
        max_length = configs.TEXT_TASKS_MAX_LENGTH[dataset_name]
        # After loading and before renaming to "labels"

        label_map = {0: 2, 1: 1, 2: 0}

        def preprocess(batch):
            tokenized = dict(
                tokenizer(
                    batch["premise"],
                    batch["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
            )
            tokenized["labels"] = [label_map[label] for label in batch["label"]]
            return tokenized

        validation_key = "validation_matched"
        encoded_dataset = raw_dataset[validation_key].map(preprocess, batched=True)
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


        # Use "validation_matched" split for consistency
        data_loader = torch.utils.data.DataLoader(
            encoded_dataset,
            batch_size=batch_size,
            shuffle=shuffle_dataset,
        )

        if inject_on_corr_preds:
            del data_loader
            df = pd.read_csv(f"data/{model_name}-{dataset_name}-predictions.csv")
            df["confidence"] = df["top1_prob"] - df["top2_prob"]
            df = df.sort_values(by="confidence", ascending=True)
            df = df[df["true_label"] == df["pred_class"]]
            if nsamples > 0:
                df = df.head(nsamples)
            indices = df.index.tolist()
            subset = encoded_dataset.select(indices)

            data_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle_dataset,
            )
            logger.info(f"{len(subset)} correct predictions found.")
            logger.info("Injecting faults on correct predictions only.")

    elif model_name == configs.GPT2:
        logger.debug("Text model selected.")
        # ----------------------------
        # Support SST-2 and MNLI
        # ----------------------------
        if dataset_name not in [configs.GLUE_SST2, configs.GLUE_MNLI]:
            raise ValueError(f"Dataset {dataset_name} not supported for text models.")

        dataset, task = dataset_name.split("_")
        test_set = load_dataset(dataset, task)

        model_dir = f"./gpt2-{task}-finetuned"
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

        num_labels = configs.TEXT_TASKS_NUM_LABELS[dataset_name]
        model = GPT2ForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

        model_for_fault = GPT2ForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
        model_for_fault.config.pad_token_id = tokenizer.pad_token_id
        model_for_fault.resize_token_embeddings(len(tokenizer))

        # ----------------------------
        # Task-specific preprocessing
        # ----------------------------
        max_length = configs.TEXT_TASKS_MAX_LENGTH[dataset_name]

        if task == "sst2":
            logger.debug("SST-2 task selected.")
            def preprocess(example):
                return tokenizer(
                    example["sentence"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )

        elif task == "mnli":
            logger.debug("MNLI task selected.")
            # MNLI has 'premise' and 'hypothesis' fields
            def preprocess(example):
                return tokenizer(
                    example["premise"],
                    example["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )

        encoded_dataset = test_set.map(preprocess, batched=True)
        encoded_dataset = encoded_dataset.rename_column("label", "labels")
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # MNLI has multiple validation splits: 'validation_matched' and 'validation_mismatched'
        if task == "mnli":
            validation_key = "validation_matched"
        else:
            validation_key = "validation"

        data_loader = torch.utils.data.DataLoader(
            encoded_dataset[validation_key],
            batch_size=batch_size,
            shuffle=shuffle_dataset,
        )

        if inject_on_corr_preds:
            del data_loader
            df = pd.read_csv(f"data/{model_name}-{dataset_name}-predictions.csv")
            df["confidence"] = df["top1_prob"] - df["top2_prob"]
            df = df.sort_values(by="confidence", ascending=True)
            df = df[df["true_label"] == df["pred_class"]]
            if nsamples > 0:
                df = df.head(nsamples)
            indices = df.index.tolist()
            subset = encoded_dataset[validation_key].select(indices)

            data_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle_dataset,
            )
            logger.info(f"{len(subset)} correct predictions found.")
            logger.info("Injecting faults on correct predictions only.")

    if nsamples > 0:
        subset_indices = list(range(nsamples))
        if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
            subset = torch.utils.data.Subset(test_set, subset_indices)
        # elif model_name in configs.TEXT_MODELS:
        #     if task == "mnli":
        #         subset_split = "validation_matched"
        #     else:
        #         subset_split = "validation"

        #     subset = encoded_dataset[subset_split].select(subset_indices)
        
            data_loader = torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle_dataset,
            )
            logger.info(f"Using only {nsamples} samples for injection.")

    dummy_input = None
    if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        dummy_input, _ = next(iter(data_loader))
    elif model_name in configs.TEXT_MODELS:
        for x in data_loader:
            dummy_input = {k: v for k, v in x.items() if k != "labels"}
            break

    fault_model = None
    if model_name in configs.VIT_CLASSIFICATION_CONFIGS:
        fault_model = statistical_fi.get_fault_model(
            configs.FAULT_MODEL_FILE, model_name, microop, precision, fault_model_threshold
        )
    elif model_name in configs.TEXT_MODELS:
        fault_model_mapping = {
            configs.GPT2_BLOCK: configs.BLOCK,
            configs.GPT2_ATTENTION: configs.ATTENTION,
            configs.GPT2_MLP: configs.MLP,
            configs.BART_ENCODER: configs.BLOCK,
            configs.BART_DECODER: configs.BLOCK,
            configs.BART_SDPA_ATTENTION: configs.ATTENTION,
            configs.BART_MLP: configs.MLP,
        }
        mapped_microop = fault_model_mapping.get(microop, microop)
        fault_model = statistical_fi.get_fault_model(
            configs.FAULT_MODEL_FILE, configs.VIT_BASE_PATCH16_224, mapped_microop, precision, fault_model_threshold
        )

    if fault_model is None:
        raise ValueError("Fault model not found.")
    # print("before converting to list")
    # data_loader = list(data_loader) 
    # data_loader = [data_loader[0], data_loader[-1]]
    # print(f"Data loader length: {len(data_loader)}")
    # exit(0)
    if fault_model.empty:
        raise ValueError("Fault model not found.")

    hook, handler, range_restrict_handlers = statistical_fi.hook_microop(
        model_for_fault,
        model_name,
        microop,
        batch_size,
        len(subset),
        fault_model,
        dummy_input,
        target_layer,
        injection_type,
        seed=seed,
        bit_position=bitflip_position,
        dataset_name=dataset_name,
        range_restriction_mode=range_restriction_mode,
    )

    if args.load_critical:
        hook.set_critical_batches(batch_indices)
        hook.set_save_critical_logits(save_critical_logits)
    del dummy_input

    logger.info(f"Injecting on {len(data_loader)} batches of size {batch_size}...")

    injection_type_for_file = injection_type
    if injection_type == statistical_fi.InjectionType.SINGLE:
        injection_type_for_file = f"{str(injection_type)}_bit{bitflip_position}"
    result_file = result_data_utils.get_result_filename(
        model_name,
        dataset_name,
        precision,
        microop,
        fault_model_threshold,
        seed,
        target_layer,
        injection_type_for_file,
        specific_seed,
        range_restriction_mode,
    )
    result_df = result_data_utils.init_result_data(
        configs.RESULTS_DIR, result_file, configs.RESULT_COLUMS
    )

    logger.info("Running injections...")
    if save_top5prob:
        get_faulty_top5(
            model_name,
            microop,
            model_for_fault,
            data_loader,
            precision,
            device,
            batch_size,
            logger,
        )
    else:
        if model_name in configs.TEXT_MODELS:
            run_injections_gpt2(
                model_name,
                dataset_name,
                microop,
                model,
                model_for_fault,
                data_loader,
                precision,
                device,
                batch_size,
                result_df,
                result_file,
                logger,
            )
        else:
            run_injections(
                model_name,
                dataset_name,
                microop,
                model,
                model_for_fault,
                data_loader,
                precision,
                device,
                batch_size,
                result_df,
                result_file,
                logger,
            )

    # torch.save(hook.get_relative_errors(), f"data/rel_err_outputs2/{model_name}-{str(target_layer)}-{seed}-rel_err.pt")

    handler.remove()
    for rr_handler in range_restrict_handlers:
        rr_handler.remove()


if __name__ == "__main__":
    main()
