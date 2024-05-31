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
from torch.optim.lr_scheduler import ReduceLROnPlateau

from typing import Dict


def main():
    # ignore all the warnings for better visualization of the training process
    ignore_warnings()

    print(f"[INFO] Setting experiment")
    experiment_name: str = "Early Stopping and Save Best Weights"  # the name of the experiment
    artifact_path: str = "Main_Artifacts"  # name of the folder where the artifacts will be saved

    # set the experiment and print out the experiment_id
    experiment_id = get_and_set_experiment(
        experiment_name=experiment_name,
        artifact_path=artifact_path,
        tags={
            "Model": "GAN - Generative Adversarial Networks",
            "Purpose": "Apply Early Stopping and Save best weights",
        }
    )
    print(f"Experiment ID: {experiment_id}")

    # create a params dictionary, this will the use for later to log all the params needed
    params: Dict[str, float | int | str] = {
        "batch_size": 128,  # the first to create is the batch size
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
    params["learning_rate"] = 0.0002
    params["num_epochs"] = 500

    # create and tracking what loss function has been used
    print(f"[INFO] Setting Loss and Optimizers")
    loss_function = nn.BCELoss()
    params["loss_function"] = loss_function.__class__.__name__  # .__class__.__name__ will return the name of the class

    # create and tracking optimizer for discriminator
    optimizer_discriminator = torch.optim.RMSprop(
        discriminator.parameters(),
        lr=params["learning_rate"]
    )
    params["discriminator_optimizer"] = optimizer_discriminator.__class__.__name__

    # create and tracking optimizers for generator
    optimizer_generator = torch.optim.RMSprop(
        generator.parameters(),
        lr=params["learning_rate"]
    )
    params["generator_optimizer"] = optimizer_generator.__class__.__name__

    print(f"[INFO] Implementing Early Stopping and Learning Rate Scheduler")
    # setting up early stopping
    params["early_stopping_patience"] = 100
    best_discriminator_loss = float("-inf")
    best_generator_loss = float("inf")
    prev_gen_loss = float("inf")
    best_model_state_dict = None
    num_epochs_without_improvement = 0

    # setting up learning rate scheduler for the discriminator
    params["d_factor"] = .5
    params["d_mode"] = "max"
    params["d_lr_patience"] = 5
    params["d_min_lr"] = 1e-15
    d_scheduler = ReduceLROnPlateau(
        optimizer_discriminator,
        mode=params["d_mode"],
        factor=params["d_factor"],
        patience=params["d_lr_patience"],
        min_lr=params["d_min_lr"],
        verbose=True
    )
    params["d_lr_scheduler"] = d_scheduler.__class__.__name__

    # setting up learning rate scheduler for the generator
    params["g_factor"] = .5
    params["g_mode"] = "min"
    params["g_lr_patience"] = 5
    params["g_min_lr"] = 1e-15
    g_scheduler = ReduceLROnPlateau(
        optimizer_generator,
        mode=params["g_mode"],
        factor=params["g_factor"],
        patience=params["g_lr_patience"],
        min_lr=params["g_min_lr"],
        verbose=True
    )
    params["g_lr_scheduler"] = g_scheduler.__class__.__name__

    print(f"[INFO] Implementing the training process")
    # this is for tracking the record of the training process, use for plotting and tracking the curves
    history = {
        "discrimination_loss": list(),
        "generator_loss": list()
    }

    with mlflow.start_run(
            experiment_id=experiment_id,  # pass in the experiment_id
            log_system_metrics=True  # track the hardware usage during training
    ) as run:

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

        # log all the params in the params dictionary
        mlflow.log_params(params=params)

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

            # update the learning rate
            g_scheduler.step(loss_generator)
            d_scheduler.step(loss_discriminator)
            curr_d_lr = optimizer_discriminator.param_groups[0]['lr']
            curr_g_lr = optimizer_generator.param_groups[0]['lr']
            print(f"Discriminator LR: {curr_d_lr}")
            print(f"Generator LR: {curr_g_lr}")
            mlflow.log_metric("discriminator_lrs", curr_d_lr, step=epoch)
            mlflow.log_metric("generator_lrs", curr_g_lr, step=epoch)

            # saved the losses for plotting
            history["discrimination_loss"].append(loss_discriminator.item())
            history["generator_loss"].append(loss_generator.item())

            # log the losses in each epoch
            mlflow.log_metric("discrimination_loss", loss_discriminator, step=epoch)
            mlflow.log_metric("generator_loss", loss_generator, step=epoch)

            # save generated samples by the generator every 5 epochs
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

            # check if the discriminator loss increased while generator loss decreased
            if loss_discriminator.item() > best_discriminator_loss:
                # save the highest loss for discriminator
                best_discriminator_loss = loss_discriminator.item()

            if loss_generator.item() < best_generator_loss:
                #  save the lowest for generator
                best_generator_loss = loss_generator.item()

                # save the lowest loss for generator
                best_model_state_dict = generator.state_dict()

                # reset the number of epochs without improvement
                num_epochs_without_improvement = 0
            else:
                # increase the number of epochs without improvement
                num_epochs_without_improvement += 1

            # break the for loop if the number of epochs surpass the early stopping patience
            if num_epochs_without_improvement > params["early_stopping_patience"]:
                print("Training Process Stopped Early")
                break
            print(f"Num Epoch Without Improvement: {num_epochs_without_improvement}/{params['early_stopping_patience']}")

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
            "This is an experiment for using early stopping and how to track it using mlflow"
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

        # tracking best choosen metrics
        mlflow.log_metrics({
            "highest_dis_loss": best_discriminator_loss,
            "lowest_gen_loss": best_generator_loss
        })

        if best_model_state_dict is not None:
            # track the lowest loss version of the generator
            best_generator = Generator(num_noise=params["num_noise"]).to(device=device)
            best_generator.load_state_dict(best_model_state_dict)
            # log the generator
            mlflow.pytorch.log_model(
                pytorch_model=best_generator,
                artifact_path="LowestLossWeights/model/" + generator.__class__.__name__,
                signature=model_signature
            )

            check_model_figs = figs_generated(
                device=device,
                generator=best_generator,
                num_noise=params["num_noise"],
                num_images=10
            )
            mlflow.log_figure(
                figure=check_model_figs,
                artifact_file="LowestLossWeights/generated_images/plots.png"
            )


if __name__ == "__main__":
    main()
