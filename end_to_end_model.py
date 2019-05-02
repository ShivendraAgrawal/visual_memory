import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from cnn_model import Net
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=1024, batch_size=3):
        super(LSTM, self).__init__()
        self.tags = {'forward':0, 'left':1, 'right':2}

        self.nb_lstm_layers = nb_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # don't count the padding tag for the classifier output
        self.nb_tags = len(self.tags)

        # # when the model is bidirectional we double the output dimension
        # self.lstm

        self.embedding = Net(self.embedding_dim)
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)


    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if torch.cuda.is_available():
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        print(X.size())
        batch_size, seq_len, C, H, W = X.size()
        c_in = X.view(batch_size * seq_len, C, H, W)
        print(c_in.shape)
        c_out = self.embedding(c_in)
        print(c_out.shape)
        r_in = c_out.view(batch_size, seq_len, -1)
        print(r_in.shape)
        # --------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        # X = self.embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # X = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        # now run through LSTM
        r_out, self.hidden = self.lstm(r_in, self.hidden)
        print(r_out.shape)

        # # undo the packing operation
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # X = X.contiguous()

        # run through actual linear layer
        print(r_out[:, -1, :])
        tag_space= self.hidden_to_tag(r_out[:, -1, :])

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        tag_scores = F.log_softmax(tag_space, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, self.nb_tags)

        # Y_hat = X
        return tag_scores


model = LSTM(1)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.003)

def load_dataset(batch_size):
    X=np.load("./movo_data/images.npy")
    y=np.load("./movo_data/labels.npy")

    print(X.shape, y.shape)

    X=torch.tensor(X)
    y=torch.tensor(y)


    dataset = TensorDataset(X.float(),y.float())

    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    return loader



def train(epoch):
    batch_size = 3
    train_loader = load_dataset(batch_size)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        # data = np.expand_dims(data, axis=1)
        data = torch.FloatTensor(data)
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


torch.manual_seed(1)
num_epochs =5
for epoch in range(1, num_epochs + 1):
    train(epoch)
    # test()

