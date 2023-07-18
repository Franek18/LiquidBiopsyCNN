import torch
from torch import nn
# Data format - 180x134, 24089 active genes and 31 nonactive genes (black genes)


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=10, stride=2)
        # After conv output = (32, 86, 63)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        # After maxpool output = (32, 43, 31)
        self.flatten = nn.Flatten()
        # After flatten output = (42656,)
        self.fc1 = nn.Linear(in_features=42656, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # print('x shape before conv = ', x.shape)
        z = self.conv(x)
        # print('z shape after conv = ', z.shape)
        z = self.relu(z)
        z = self.maxpooling(z)
        # print('z shape after maxpool = ', z.shape)
        z = self.flatten(z)
        # print('z shape after flatten = ', z.shape)
        z = self.fc1(z)
        # print('z shape after fc1 = ', z.shape)
        z = self.fc2(z)
        # print('z shape after fc2 = ', z.shape)
        out = self.softmax(z)

        return out


class FlatCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=134, stride=134)
        # After conv output = (32, 179)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(kernel_size=2, stride=2)
        # After maxpool output = (32, 89)
        self.flatten = nn.Flatten()
        # After flatten output = (2848,)
        self.fc1 = nn.Linear(in_features=2848, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        z = self.conv(x)
        # print(f'z after conv = ', z.shape)
        z = self.relu(z)
        z = self.maxpooling(z)
        # print(f'z after maxp = ', z.shape)
        z = self.flatten(z)
        z = self.fc1(z)
        z = self.fc2(z)
        out = self.softmax(z)

        return out


class HybridCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 134), stride=1)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten1 = nn.Flatten()

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(180, 1), stride=1)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.flatten2 = nn.Flatten()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=5024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # print("x.shape = ", x.shape)
        z1 = self.conv1(x)
        # print("z1.shape = ", z1.shape)
        z1 = self.relu(z1)
        z1 = self.maxpooling1(z1)
        z1 = self.flatten1(z1)

        z2 = self.conv2(x)
        z2 = self.relu(z2)
        z2 = self.maxpooling2(z2)
        z2 = self.flatten2(z2)

        z = torch.cat((z1, z2), dim=1)
        # print("z after concatenation = ", z.shape)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        out = self.softmax(z)

        return out

