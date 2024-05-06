import mlflow

from models.discriminator import Discriminator
from models.generator import Generator

from dataset.dataset_utils import data_loader
from dataset.dataset_utils import get_classes

from training.training_utils import train

from utils.mlflow_utils import get_and_set_experiment
from utils.mlflow_utils import get_performance_plots
from utils.mlflow_utils import get_log_inputs
from utils.mlflow_utils import get_signature

from utils.utils import ignore_warnings
from utils.utils import figs_generated
from utils.utils import set_up_pytorch

import torch
from torch import nn

from typing import Dict


def main():
    # ignore all the warnings for better visualization of the training process
    ignore_warnings()

    print(f"[INFO] Setting experiment")
    experiment_name: str = "Generative Adversarial Networks"  # the name of the experiment
    artifact_path: str = "GAN"  # name of the folder where the artifacts will be saved

    # set the experiment and print out the experiment_id
    experiment_id = get_and_set_experiment(
        experiment_name=experiment_name,
        artifact_path=artifact_path,
        tags={
            "Model": "GAN - Generative Adversarial Networks",
            "Purpose": "Creating capitalized letters",
        }
    )
    print(f"Experiment ID: {experiment_id}")

    # create a params dictionary, this will the use for later to log all the params needed
    params: Dict[str, float | int | str] = {
        "batch_size": 16,  # the first to create is the batch size
    }

    # create dataloader
    print(f"[INFO] Getting Dataloader")
    train_loader = data_loader(
        csv_file="dataset/data/A_Z Handwritten Data.csv",
        batch_size=params["batch_size"]
    )

    # set up the random seed for regeneration of the project purpose
    print(f"[INFO] Setting up PyTorch")
    # add the random seed for informative purpose
    params["seed"] = 111
    params["device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    device = set_up_pytorch(seed=params["seed"])

    print(f"[INFO] Implementing the Discriminator")
    discriminator = Discriminator().to(device=device)

    print(f"[INFO] Implementing the Generator")
    params["num_noise"] = 100  # tracking the number of noise

    generator = Generator(num_noise=params["num_noise"]).to(device=device)

    # tracking the learning rate and epochs
    params["learning_rate"] = 0.0001
    params["num_epochs"] = 50

    # create and tracking what loss function has been used
    print(f"[INFO] Setting Loss and Optimizers")
    loss_function = nn.BCELoss()
    params["loss_function"] = loss_function.__class__.__name__  # .__class__.__name__ will return the name of the class

    # create and tracking optimizer for discriminator
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(),
        lr=params["learning_rate"]
    )
    params["discriminator_optimizer"] = optimizer_discriminator.__class__.__name__

    # create and tracking optimizers for generator
    optimizer_generator = torch.optim.Adam(
        generator.parameters(),
        lr=params["learning_rate"]
    )
    params["generator_optimizer"] = optimizer_generator.__class__.__name__

    print(f"[INFO] Implementing the training process")
    # this is for tracking the record of the training process, use for plotting and tracking the curves
    history = {
        "discrimination_loss": list(),
        "generator_loss": list()
    }

    with mlflow.start_run(
            experiment_id=experiment_id,  # pass in the experiment_id
            log_system_metrics=False  # do not track the hardware usage during training
    ) as run:

        for epoch in range(params["num_epochs"]):
            print(f"Epoch: {epoch}")

            # use the train function to train the model and get the losses
            loss_discriminator, loss_generator = train(
                train_loader=train_loader,
                device=device,
                generator=generator,
                discriminator=discriminator,
                optimizer_generator=optimizer_generator,
                optimizer_discriminator=optimizer_discriminator,
                loss_function=loss_function,
                num_noise=params["num_noise"],
            )

            # saved the losses for plotting
            history["discrimination_loss"].append(loss_discriminator.item())
            history["generator_loss"].append(loss_generator.item())

            # log the losses in each epoch
            mlflow.log_metric("discrimination_loss", loss_discriminator, step=epoch)
            mlflow.log_metric("generator_loss", loss_generator, step=epoch)

            # save 16 generated samples by the generator every 5 epochs
            if epoch % 5 == 0:
                plot_fig = figs_generated(
                    device=device,
                    generator=generator,
                    num_noise=params["num_noise"]
                )
                mlflow.log_figure(
                    figure=plot_fig,
                    artifact_file=f"{artifact_path}/artifacts/generated_images/epoch{epoch}.png"
                )
            print(f"DiscriminationLoss: {loss_discriminator} | GeneratorLoss: {loss_generator}")

        # log all the params in the params dictionary
        mlflow.log_params(params=params)

        # add some informative tags
        mlflow.set_tags(
            {
                "type": "GAN",
                "dataset": "HandwrittenAZDataset",
            }
        )

        # add a brief description
        mlflow.set_tag(
            "mlflow.note.content",
            "This is a GAN model for generating capitalized handwritten letters"
        )

        # log the input dataset
        input_df = get_log_inputs(
            csv_file="dataset/data/A_Z Handwritten Data.csv",
            dataset_name="A_Z Handwritten Data"
        )
        mlflow.log_input(dataset=input_df)

        # log the plots using the history
        figure = get_performance_plots(history=history)
        mlflow.log_figure(
            figure=figure,
            artifact_file=f"{artifact_path}/artifacts/curves/loss_curves.png"
        )

        # log the model summary
        with open("models/discriminator.txt", "w") as file:
            file.write(str(discriminator))
        with open("models/generator.txt", "w") as file:
            file.write(str(generator))
        mlflow.log_artifact(
            local_path="models/discriminator.txt",
            artifact_path=artifact_path+"/artifacts/model_summary"
        )
        mlflow.log_artifact(
            local_path="models/generator.txt",
            artifact_path=artifact_path+"/artifacts/model_summary"
        )

        # create discriminator's model signature
        inputs = next(iter(train_loader))[0]
        model_signature = get_signature(
            inputs=inputs,
            device=device,
            model=discriminator
        )
        # log the discriminator
        mlflow.pytorch.log_model(
            pytorch_model=discriminator,
            artifact_path=artifact_path + "/project/models" + discriminator.__class__.__name__,
            signature=model_signature
        )

        # create generator's model signature
        inputs = torch.randn((params["batch_size"], params["num_noise"]))
        model_signature = get_signature(
            inputs=inputs,
            device=device,
            model=generator
        )
        # log the generator
        mlflow.pytorch.log_model(
            pytorch_model=generator,
            artifact_path=artifact_path + "/project/models" + generator.__class__.__name__,
            signature=model_signature
        )


if __name__ == "__main__":
    main()
