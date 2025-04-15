import pandas as pd
import os
from statistical_fi import LayerChoice, InjectionType


def get_result_filename(
    model_name: str,
    dataset_name: str,
    precision: str,
    microop: str,
    float_threshold_FM: str,
    seed: int,
    layer: LayerChoice,
    it: InjectionType,
) -> str:
    return f"{model_name}_{dataset_name}_{precision}_{microop}_{float_threshold_FM}_{seed}_layer-{str(layer)}_it-{str(it)}.csv"


def init_result_folder(data_path: str) -> None:
    path = os.path.abspath(data_path)
    if not os.path.exists(path):
        os.makedirs(path)


def init_result_data(data_path: str, result_file: str, columns) -> pd.DataFrame:
    init_result_folder(data_path)

    result_file_path = os.path.join(data_path, result_file)
    if os.path.exists(result_file_path):
        os.remove(result_file_path)

    df = []
    return df


def append_row(
    df,
    groundtruth,
    prediction_without_fault,
    prediction_with_fault,
    top1_wo_fault,
    top2_wo_fault,
    top1_w_fault,
    top2_w_fault,
) -> pd.DataFrame:
    df.append(
        {
            "ground_truth": groundtruth,
            "prediction_without_fault": prediction_without_fault,
            "prediction_with_fault": prediction_with_fault,
            "top1_wo_fault": top1_wo_fault,
            "top2_wo_fault": top2_wo_fault,
            "top1_w_fault": top1_w_fault,
            "top2_w_fault": top2_w_fault,
        }
    )
    return df


def save_result_data(result_df: pd.DataFrame, data_path: str, result_file: str) -> None:
    result_file_path = os.path.join(data_path, result_file)
    result_df.to_csv(result_file_path, index=False)
