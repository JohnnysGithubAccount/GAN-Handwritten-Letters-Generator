import warnings
import torch
import numpy as np
import os
from torch import nn
import matplotlib.pyplot as plt


def ignore_warnings() -> None:
    """
    ignore warnings for better visualization of the training process
    :return:
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
    # this just for enabling logging system metrics if wanted
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


def set_up_pytorch(seed: int = 42) -> str:
    """
    set up random seed and get the current device
    :param seed: the seed you wanted to set
    :return: cuda or cpu
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def figs_generated(device: str,
                   generator: nn.Module,
                   num_noise: int,
                   num_images: int = 6) -> plt.Figure:
    """
    get generated images by the generator
    :param device: cpu or gpu
    :param generator: generator model
    :param num_noise: number of noise
    :param num_images: how many images per row and per column
    :return: a figure
    """
    latent_space_samples = torch.randn(num_images * num_images, num_noise).to(device=device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()

    fig, axs = plt.subplots(num_images, num_images, figsize=(12, 12))  # Changed to 12x12 grid
    # Display total of (num_images  * num_images) images/ default 36 images
    for i in range(num_images * num_images):
        ax = axs[i // num_images, i % num_images]
        ax.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])

    return fig


def main():
    pass


if __name__ == "__main__":
    main()
