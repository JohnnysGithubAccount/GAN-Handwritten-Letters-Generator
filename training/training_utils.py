import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm


def train(
        train_loader: DataLoader,
        device: str,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_discriminator: torch.optim,
        optimizer_generator: torch.optim,
        loss_function,
        num_noise: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    perform training in each training epoch
    :param train_loader: data loader
    :param device: using gpu or cpu
    :param generator: generator model
    :param discriminator: discriminator model
    :param optimizer_discriminator: optimizer for discriminator
    :param optimizer_generator: optimizer for generator
    :param loss_function: loss function
    :param num_noise: number of noise for the generator
    :return: losses of discriminator and generator
    """

    # set the model to training model
    discriminator.train()
    generator.train()

    # loop through all the batch
    for batch, (real_samples, mnist_labels) in tqdm(enumerate(train_loader)):
        # set how much sample, make it equal the batch size
        num_sample = real_samples.shape[0]

        # Data for training the discriminator
        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((num_sample, 1)).to(
            device=device
        )  # label all the real samples data to 1

        latent_space_samples = torch.randn((num_sample, num_noise)).to(
            device=device
        )  # create noise for the generator
        generated_samples = generator(latent_space_samples)  # generate images
        generated_samples_labels = torch.zeros((num_sample, 1)).to(
            device=device
        )  # label all the generated samples to 0

        # combine the real and the fake samples to make the training data for discriminator
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()  # clear the gradient
        # use the discriminator to classify the real and the fake samples
        output_discriminator = discriminator(all_samples)
        # calculate the discriminator loss
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels
        )
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((num_sample, num_noise)).to(
            device=device
        )

        # Training the generator
        generator.zero_grad()  # clear the gradient
        generated_samples = generator(latent_space_samples)  # generate new samples
        # use the discriminator to classify only on the new generated set
        output_discriminator_generated = discriminator(generated_samples)
        # calculate the loss for generator
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

    return loss_discriminator, loss_generator


def validate():
    pass


