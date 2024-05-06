import mlflow
from mlflow.models.signature import infer_signature
from mlflow.data.pandas_dataset import from_pandas
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import pandas as pd
import torch


def get_and_set_experiment(
        experiment_name: str,
        artifact_path: str,
        tags: Dict[str, str]) -> str:
    """
    set the experiment for tracking using mlflow
    :param experiment_name:name of the experiment
    :param artifact_path: the name of the path that will save the artifacts
    :param tags: informative key: value format
    :return: experiment id
    """
    try:
        # try to create an experiment with the given name
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_path,
            tags=tags
        )
    except:
        # if the experiment already exist, get the experiment id
        experiment_id = mlflow.get_experiment_by_name(
            name=experiment_name
        ).experiment_id
    finally:
        # finally, set the experiment using the experiment name
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_performance_plots(
        history: Dict[str, list]) -> plt.Figure:
    """
    get the loss and accuracy curves of the model to track using mlflow
    :param history: dictionary of training history
    :return: dictionary of curves
    """
    loss_curve = plt.figure()
    plt.plot(
        np.arange(len(history["discrimination_loss"])),
        history["discrimination_loss"],
        label="Discrimination Loss"
    )
    plt.plot(
        np.arange(len(history["generator_loss"])),
        history["generator_loss"],
        label="Generator Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    return loss_curve


def get_signature(
        inputs: torch.Tensor,
        device: str,
        model: torch.nn.Module) -> mlflow.models.ModelSignature:

    inputs = inputs.to(device)
    outputs = model(inputs)  # predict from a sample input to get the output format

    model_signature = infer_signature(
        inputs.cpu().detach().numpy(),
        outputs.cpu().detach().numpy()
    )  # get the model signature

    return model_signature


def get_log_inputs(
        csv_file: str,
        dataset_name: str) -> mlflow.data.Dataset:
    """
    create a dataset object of mlflow for tracking dataset
    :param csv_file: path to the csv file
    :param dataset_name: the name of the dataset
    :return:
    """
    # load the data from csv file
    df = pd.read_csv(csv_file)
    df = from_pandas(
        df=df,
        source=csv_file,
        name=dataset_name,
    )  # get mlflow dataset object from a pandas dataframe
    return df


def main():
    pass


if __name__ == "__main__":
    main()
