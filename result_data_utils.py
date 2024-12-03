import pandas as pd
import os

def get_result_filename(model_name: str, dataset_name: str, precision: str, float_threshold_FM: str) -> str:
    return f"{model_name}_{dataset_name}_{precision}_{float_threshold_FM}.csv"

def init_result_folder(data_path: str) -> None:
    path = os.path.abspath(data_path)
    if not os.path.exists(path):
        os.makedirs(path)

def init_result_data(data_path: str, result_file: str, columns) -> pd.DataFrame:
    init_result_folder(data_path)
    
    result_file_path = os.path.join(data_path, result_file)
    if os.path.exists(result_file_path):
        os.remove(result_file_path)

    df = pd.DataFrame(columns=columns)
    return df

def append_row(df, model_name, dataset, precision, microop, groundtruth, prediction_without_fault, prediction_with_fault):
    df = df.append({
        "model": model_name,
        "dataset": dataset,
        "precision": precision,
        "microop": microop,
        "ground_truth": groundtruth,
        "prediction_without_fault": prediction_without_fault,
        "prediction_with_fault": prediction_with_fault
    }, ignore_index=True)
    return df

def save_result_data(result_df: pd.DataFrame, data_path: str, result_file: str) -> None:
    result_file_path = os.path.join(data_path, result_file)
    result_df.to_csv(result_file_path, index=False)