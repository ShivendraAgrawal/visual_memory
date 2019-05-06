import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class CNN(nn.Module):
    def __init__(self, feature_length):
        super(CNN, self).__init__()
        # input image size - 128*128*3

        # # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7020, 1000)
        self.fc2 = nn.Linear(1000, feature_length)


    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x=self.pool(self.pool(self.pool(x)))
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1,7020 )

        path_abs_input = F.relu(self.fc1(x))
        path_abs_input = self.fc2(path_abs_input)

        return path_abs_input