import torch
import torch.nn as nn


class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SiameseNetwork(nn.Module):

    def __init__(self, features=False):
        super(SiameseNetwork, self).__init__()

        self.features = features

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),

            Flatten(),
            nn.Linear(65536, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        output = self.cnn(x)
        return output


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        if self.features:
            return output1, output2
        else:
            #Computes L1 distance
            output = torch.sum(torch.abs((output1 - output2)), 1)
            #passing the distance to sigmoid layer
            #output = self.fc(output)
            return output

