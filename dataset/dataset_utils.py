import matplotlib.pyplot as plt
import string
import pandas as pd
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# create a class Dataset for loading data
class HandwrittenAZDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 transform: transforms = None):
        self.dataframe = df
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # the image is the idx row, from column 1 till end
        image = self.dataframe.iloc[idx, 1:].values.astype('uint8').reshape((28, 28, 1))
        # the label is the first column of the idx row
        label = self.dataframe.iloc[idx, 0]
        # apply transform
        if self.transform:
            image = self.transform(image)

        return image, label


def loading_data(
        csv_file: str) -> Tuple[Dataset, Dataset] | Dataset:
    """
    create a dataset object from the csv file
    :param csv_file: the path to the csv file dataset
    :return: a dataset object
    """

    # load the csv ad a dataframe
    az_data = pd.read_csv(csv_file)

    # transform for the image
    transform = transforms.Compose([
        transforms.ToTensor(),  # change it to tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the data to the range [-1, 1]
    ])

    # return the dataset object
    return HandwrittenAZDataset(
        df=az_data,
        transform=transform
    )


def data_loader(
        csv_file: str,
        batch_size: int = 32,) -> DataLoader:
    """
    create and return a dataloader
    :param csv_file: the path to the csv file dataset
    :param batch_size: the batch size
    :return: a dataloader
    """

    # get the dataset object
    train_data = loading_data(csv_file=csv_file)

    # create a dataloader object
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    return train_loader


def get_classes() -> List[str]:
    return [char for char in string.ascii_uppercase]


def plot_images(
        dataloader: DataLoader,
        class_names: List[str]) -> None:
    """
    use for plotting images for better visualization of the dataset
    :param dataloader: data loader
    :param class_names: class names
    :return: None
    """
    # set the random seed to have the same initial everytime
    torch.manual_seed(42)

    # setup number of images
    side = 4
    num_images = side * side

    # get a batch of images and labels
    images, labels = next(iter(dataloader))

    # plotting images
    fig, axes = plt.subplots(side, side, figsize=(9, 9))
    axes = axes.flatten()

    for i in range(num_images):
        rand_idx = torch.randint(0, len(images), size=[1]).item()

        img, label = images[rand_idx].numpy().squeeze(), labels[rand_idx].item()

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(class_names[label])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    train_loader = data_loader(
        csv_file="data/A_Z Handwritten Data.csv",
        batch_size=32
    )
    class_names = get_classes()
    plot_images(train_loader, class_names)


if __name__ == "__main__":
    main()
