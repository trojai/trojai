import torch.nn as nn

"""
Defines various architectures useful for the MNIST dataset.
"""


class ModdedLeNet5Net(nn.Module):
    """
    A modified LeNet architecture that seems to be easier to embed backdoors in than the network from the original
    badnets paper
    Input - (1 or 3)x28x28
    C1 - 6@28x28 (5x5 kernel)
    ReLU
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    ReLU
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, channels=1):
        super(ModdedLeNet5Net, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class BadNetExample(nn.Module):
    """
    Mnist network from BadNets paper
    Input - 1x28x28
    C1 - 1x28x28 (5x5 kernel) -> 16x24x24
    ReLU
    S2 - 16x24x24 (2x2 kernel, stride 2) Subsampling -> 16x12x12
    C3 - 16x12x12 (5x5 kernel) -> 32x8x8
    ReLU
    S4 - 32x8x8 (2x2 kernel, stride 2) Subsampling -> 32x4x4
    F6 - 512 -> 512
    tanh
    F7 - 512 -> 10 Softmax (Output)
    """

    def __init__(self):
        super(BadNetExample, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=-1)
        )

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
