import mlflow.pytorch as mlflow_pytorch
from utils.utils import set_up_pytorch, figs_generated
import matplotlib.pyplot as plt


def main():

    num_images = 5
    num_noise = 100
    device = set_up_pytorch(111)

    # model uri in the format "models:/models_name/version(ex: latest, 1, 2...)"
    model_uri = "models:/FCGeneratorModel/1"
    generator = mlflow_pytorch.load_model(
        model_uri=model_uri
    ).to(device=device)

    fig = figs_generated(
        device=device,
        generator=generator,
        num_noise=num_noise,
        num_images=num_images
    )
    plt.show()


if __name__ == "__main__":
    main()
