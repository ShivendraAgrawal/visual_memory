import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from cnn_model import CNN
import torchvision.models as models
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


class LSTM(nn.Module):
    def __init__(self, nb_layers, nb_lstm_units=100, embedding_dim=500, batch_size=5):
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

        self.embedding = CNN(self.embedding_dim)
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
        # print(hidden_a.shape, hidden_b.shape)
        return (hidden_a, hidden_b)

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()
        # print(X.size())
        batch_size, seq_len, H, W , C= X.size()
        c_in = X.view(batch_size * seq_len, C, H, W)
        # print(c_in.shape)
        c_out = self.embedding(c_in)

        # print(c_out.shape)
        r_in = c_out.view(batch_size, seq_len, -1)
        # print(r_in.shape)
        # --------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        # X = self.embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # X = torch.nn.utils.rnn.pack_padded_sequence(x, X_lengths, batch_first=True)

        # now run through LSTM
        r_out, self.hidden = self.lstm(r_in, self.hidden)
        # print(r_out.shape)

        # # undo the packing operation
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # X = X.contiguous()

        # run through actual linear layer
        # print(r_out[:, -1, :].shape)
        r_out = r_out.contiguous().view(batch_size * seq_len,-1)
        # print(r_out.shape)
        # tag_space= self.hidden_to_tag(r_out[:, -1, :])
        tag_space = self.hidden_to_tag(r_out)
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print(tag_scores.shape)
        # tag_scores = tag_scores.view(batch_size, seq_len, -1)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, self.nb_tags)

        # Y_hat = X
        return tag_scores


def loss_plots(train_loss, valid_loss):
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['training', 'validation'], loc='lower right')

    plt.savefig("loss_plots")

def train(train_loader, valid_loader, batch_size, n_epochs):
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    model = LSTM(1, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(model)

    if train_on_gpu:
        model.cuda()

    valid_loss_min = np.Inf  # track change in validation loss

    train_loss_list = []
    valid_loss_list = []

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            target = target.long()
            data = torch.FloatTensor(data)
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            target = target.view(-1)
            # calculate the batch loss
            loss = F.nll_loss(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            target = target.long()
            data = torch.FloatTensor(data)
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            target = target.view(-1)
            # calculate the batch loss
            loss = F.nll_loss(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)*10
        valid_loss = valid_loss / len(valid_loader.dataset)*10

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'end_to_end_net.pt')
            valid_loss_min = valid_loss

    loss_plots(train_loss_list, valid_loss_list)

def save_results(PATH, loader, batch_size, seq_length):
    train_on_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()

    model = LSTM(1, batch_size=batch_size)

    model.load_state_dict(torch.load(PATH))

    model.eval()

    if train_on_gpu:
        model.cuda()


    class_correct = [0., 0., 0.]
    class_total = [0., 0., 0.]


    for data, target in loader:
        target = target.long()
        print(target.shape)
        data = torch.FloatTensor(data)
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        target = target.view(-1)
        # calculate the batch loss
        loss = F.nll_loss(output, target)

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        print("Pred target", pred, target)

        # compare predictions to true label
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

        for i in range(batch_size*seq_length):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

            # print("predicted label: ",correct[i].item())
            # print("correct label: ", label)

    # for i in range(3):
    #     print('Accuracy of {} : {} / {} = {:.4f} %'.format(i,
    #                                                        class_correct[i], class_total[i],
    #                                                        100 * class_correct[i] /
    #                                                        class_total[i]))

    print('\nAccuracy (Overall): %2d%%' % (
            100. * np.sum(class_correct) / np.sum(class_total)))


if __name__ == '__main__':

    batch_size = 1
    n_epochs = 50
    validation_split = .2
    testing_split = 0.2
    shuffle_dataset = True
    # random_seed = 42

    X = np.load("./movo_data/images_v2.npy")
    y = np.load("./movo_data/labels_v2.npy")

    print(X.shape, y.shape)
    seq_length = X.shape[1]
    print(seq_length)

    X = torch.tensor(X)
    y = torch.tensor(y)

    dataset = TensorDataset(X.float(), y.float())
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    valid_split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        # np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[valid_split:], indices[:valid_split]

    test_split = int(np.floor(testing_split * len(train_indices)))

    train_indices, test_indices = train_indices[test_split:], train_indices[:test_split]
    print(len(train_indices), len(val_indices), len(test_indices))

    print(test_indices)

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=test_sampler)

    train(train_loader, valid_loader, batch_size, n_epochs)
    print("Valid")
    save_results('end_to_end_net.pt', valid_loader, batch_size,seq_length)
    print("test")
    save_results('end_to_end_net.pt', test_loader, batch_size, seq_length)