#! /usr/bin/env python3

import os
import sys

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath(os.path.join("..", "utils")))

import configs
from cli.parsers import InputSelectionParser
import cli.logger_formatter as logger_formatter
import torch
import numpy as np
import utils.model_utils as model_utils
import utils.result_data_utils as result_data_utils
from methods import InputSelectionFactory
import pandas as pd
from variance import Variance
from dsa import DSA
from max_p import MaxP
from confidence import Confidence


def main():
    parser = InputSelectionParser()
    args = parser.parse_args()

    # Parse arguments
    precision = args.precision
    batch_size = args.batch_size
    device = args.device
    seed = args.seed
    shuffle_dataset = args.shuffle_dataset
    model_name = args.model
    dataset_name = args.dataset
    verbose = args.verbose
    load_corr_pred = args.load_correct_predictions
    input_selection_method = args.method
    min_batch = args.min_batch
    max_batch = args.max_batch

    logger = logger_formatter.logging_setup(__name__, None, False, verbose)

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Model init...")
    model = model_utils.get_model(model_name, precision)
    model = model.to(device)
    transforms = model_utils.get_vit_transforms(model, precision)

    logger.info("Dataset init...")
    test_set, data_loader = model_utils.get_dataset(
        dataset_name, transforms, batch_size, shuffle_dataset
    )

    if min_batch != 0 and max_batch != 0:
        indices = list(range(min_batch*batch_size, (max_batch-1)*batch_size, 1))
        subset = torch.utils.data.Subset(test_set, indices=indices)
        data_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=shuffle_dataset)

    logger.info(f"Validation set length: {len(data_loader)} batches.")

    train_set, train_loader = model_utils.get_train_set(
        dataset_name, transforms, batch_size, shuffle_dataset
    )
    logger.info(f"Validation set length: {len(train_loader)} batches.")

    num_classes = len(test_set.classes)
    if load_corr_pred:
        _, test_set = model_utils.get_correct_indices(
            test_set,
            f"../data/{model_name}_{dataset_name}_{precision}_correct_predictions.csv",
        )

        logger.info(f"{len(test_set)} correct predictions found.")
        data_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle_dataset
        )
        logger.info("Correclty predicted inputs loaded.")

    logger.info("Results init...")
    result_file = result_data_utils.get_input_selection_resfile(
        model_name,
        dataset_name,
        precision,
        seed,
        input_selection_method,
    )
    result_df = result_data_utils.init_result_data(
        os.path.join("..", configs.RESULTS_DIR, "input_selection"),
        result_file,
        ["variance"],
    )

    logger.info("Input selection init...")
    # input_selection = InputSelectionFactory.create(
    # input_selection_method,
    # model,
    # data_loader,
    # num_classes,
    # result_df,
    # k=configs.K,
    # device=device,
    # )

    input_selection = DSA(
        train_loader,
        data_loader,
        model,
        model_name,
        os.path.join("..", configs.RESULTS_DIR, "input_selection", "dsa"),
        device=device,
        min_batch=min_batch,
        max_batch=max_batch,
    )

    logger.info("Input selection...")
    input_selection.select_input()

    # logger.info("Saving results...")
    # df_res = pd.DataFrame(input_selection.df_res)
    # df_res.to_csv(
    #     os.path.join("..", configs.RESULTS_DIR, "input_selection", result_file),
    #     index=False,
    # )

    logger.info("Results saved.")
    logger.info("Input selection done.")
    logger.info("Exiting...")


if __name__ == "__main__":
    main()
