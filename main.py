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

TIME_MEASURE = []


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

    for i, (images, labels) in enumerate(data_loader):
        if precision == configs.FP16:
            images = images.half()
            labels = labels.half()

        start = time.time()
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

        logger.debug("-" * 80)
        logger.debug(f"Batch {i} - Microop: {microop}")
        for j in range(len(images)):
            logger.debug(
                f"Image {(i*batch_size)+j+1} - Ground truth: {labels[j]} - Prediction without fault: {out_wo_fault[j].item()} - Prediction with fault: {out_with_fault[j].item()}"
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

        TIME_MEASURE.append(time.time() - start)

        # if i == 1:
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
    global TIME_MEASURE

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

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if microop is None:
        raise ValueError("Microoperation not defined.")

    if statistical_fi.check_microop(model_name, microop) is False:
        raise ValueError(
            f"Microoperation {microop} not supported by the model {model_name}."
        )

    logger.info("Model init...")
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
        dataset_name, transforms, batch_size
    )
    if inject_on_corr_preds:
        _, subset = model_utils.get_correct_indices(
            test_set,
            f"data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv",
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
        data_loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=shuffle_dataset
        )
        logger.info("Injecting faults on correct predictions only.")

    dummy_input, _ = next(iter(data_loader))
    fault_model = statistical_fi.get_fault_model(
        configs.FAULT_MODEL_FILE, model_name, microop, precision, fault_model_threshold
    )
    if fault_model.empty:
        raise ValueError("Fault model not found.")
    hook, handler = statistical_fi.hook_microop(
        model_for_fault,
        model_name,
        microop,
        batch_size,
        len(subset),
        fault_model,
        dummy_input,
        target_layer,
        injection_type,
    )
    if args.load_critical:
        hook.set_critical_batches(batch_indices)
        hook.set_save_critical_logits(save_critical_logits)
    del dummy_input

    logger.info(f"Injecting on {len(data_loader)} batches of size {batch_size}...")

    result_file = result_data_utils.get_result_filename(
        model_name,
        dataset_name,
        precision,
        microop,
        fault_model_threshold,
        seed,
        target_layer,
        injection_type,
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

    handler.remove()

    if TIME_MEASURE is not None:
        average = mean(TIME_MEASURE)
        data = {
            "model": model_name,
            "microop": microop,
            "target_layer": str(target_layer),
            "ETA": average * len(data_loader),
        }
        logger.info(f"ETA for full pass: {average*len(data_loader):.2f}s")

        eta_path = "data/eta_swfi_rel_err_large_val.csv"

        if os.path.exists(eta_path):
            df = pd.read_csv(eta_path)
            data = pd.DataFrame([data])
            df = pd.concat([df, data])
            df.to_csv(eta_path, index=False)
        else:
            df = pd.DataFrame([data])
            df.to_csv(eta_path, index=False)


if __name__ == "__main__":
    main()
