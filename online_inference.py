import mlflow.pytorch as mlflow_pytorch
import torch

from utils.utils import set_up_pytorch, figs_generated
import matplotlib.pyplot as plt
import requests
import json
from random import randint
import numpy as np


def main():

    num_images = 5
    num_noise = 100
    set_up_pytorch(111)

    """
    __set these up in the terminal__
     mlflow server --host 0.0.0.0 --port 5000 
     mlflow models serve --model-uri models:/FCGeneratorModel/1 --no-conda
    """

    latent_space_samples = torch.randn(num_images * num_images, num_noise).tolist()

    payload = {"inputs": latent_space_samples}
    BASE_URI = " http://127.0.0.1:5000"
    headers = {"Content-Type": "application/json"}
    endpoint = BASE_URI + r"/invocations"

    r = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    print("Status code:", r.status_code)

    image = np.array(eval(r.text)['predictions'])
    print(image.shape)

    fig, axs = plt.subplots(num_images, num_images, figsize=(12, 12))
    axs = axs.flatten()

    for idx, ax in enumerate(axs):
        ax.imshow(np.squeeze(image[idx]), cmap='gray')
        ax.set_title(f"image numer: {idx}")
        ax.axis("off")
    plt.title("Images")
    plt.show()


if __name__ == "__main__":
    main()
