import torch
from torch import nn
import torchvision.models as models
# Data format - 180x134, 24089 active genes and 31 nonactive genes (black genes)


class VanillaCNN(nn.Module):
    def __init__(self, standard=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=10, stride=2)
        # After conv output = (32, 86, 63)
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        # After maxpool output = (32, 43, 31)
        self.flatten = nn.Flatten()
        # After flatten output = (42656,)
        if standard:
            self.fc1 = nn.Linear(in_features=266240, out_features=128)
        else:
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
    def __init__(self, standard=False):
        super().__init__()
        if standard:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 531), stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 134), stride=1)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 1))
        self.flatten1 = nn.Flatten()

        if standard:
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(267, 1), stride=1)
        else:
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(180, 1), stride=1)

        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.flatten2 = nn.Flatten()

        self.relu = nn.ReLU()
        if standard:
            self.fc1 = nn.Linear(in_features=12736, out_features=128)
        else:
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

def get_ResNet(dropout, no_layers=18, pretrained=False):
    '''
        This function returns a ResNet model with given numbers of layers.
        It adjusts a model for binary classification problem, adds a Dropout layer
        before the output and modifies input layer because our data
        have only one color channel, not 3 as in typical images.
        @param dropout: value of probability p of an element
        to be zeroed in Dropout layer.
        @param no_layers: number of layers in ResNet. It allows 18, 34 and 50
        layers wariants of ResNet architecture.
        @param pretrained: a bool value if we want to use an Imagenet pretrained weights.
        @return resnet: a ResNet class object.
    '''
    global device
    resnet = None
    # Choose no of layers in ResNet
    if no_layers == 34:
        resnet = models.resnet34(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )
    elif no_layers == 50:
        resnet = models.resnet50()
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )
    else:
        resnet = models.resnet18(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features
        # Here the size of each output sample is set to 2.
        resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, 2)
        )

    new_in_channels = 1

    layer = resnet.conv1

    # Creating new Conv2d layer
    new_layer = nn.Conv2d(in_channels=new_in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    bias=layer.bias)

    # Copying the weights from the old to the new layer
    new_layer.weight = nn.Parameter(layer.weight[:, :new_in_channels, :, :].clone())
    #print(new_layer.weight[1,0,1,1])
    new_layer.weight = new_layer.weight
    resnet.conv1 = new_layer

    return resnet