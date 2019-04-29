import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input image size - 128*128*3
        self.conv1 = nn.Conv2d(3, 64, 4)
        self.conv2 = nn.Conv2d(64, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(4096, 1024)

        self.fc2 = nn.Linear(1024, 512)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)




    def forward(self, x, tensor_action):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 4096)

        path_abs_input = torch.cat((x,tensor_action),1)

        path_abs_input = F.relu(self.fc1(path_abs_input))
        # add dropout layer
        path_abs_input = self.dropout(path_abs_input)
        # add 2nd hidden layer, with relu activation function
        path_abs_input = self.fc2(path_abs_input)

        return path_abs_input

