from torch import nn


class Generator(nn.Module):
    def __init__(self, num_noise: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_noise, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        # reshape the output of the model to become a batch of image
        output = output.view(x.size(0), 1, 28, 28)
        return output


def main():
    model = Generator(num_noise=100)
    print(model)


if __name__ == "__main__":
    main()
