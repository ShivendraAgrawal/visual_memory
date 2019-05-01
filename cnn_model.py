import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self, feature_length = 1024):
        super(Net, self).__init__()
        # input image size - 128*128*3
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding =1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding =1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding =1)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(8192, 4096)

        self.fc2 = nn.Linear(4096, feature_length)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)




    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print(x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        print(x.shape)
        x = x.view(-1, 8192)

        path_abs_input = F.relu(self.fc1(x))
        # add dropout layer
        path_abs_input = self.dropout(path_abs_input)
        # add 2nd hidden layer, with relu activation function
        path_abs_input = self.fc2(path_abs_input)

        return path_abs_input


