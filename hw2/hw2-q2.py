#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

IMAGE_PATH = "./images"
IMAGE_NAME = "new_image"

class CNN(nn.Module):
    
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        # NOTE: The image dimension is 28x28

        if not no_maxpool: # Implementation for Q2.1
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0)
            self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        else: # Implementation for Q2.2
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)

        # An affine transformation with 320 output features
        self.fc1 = nn.Linear(16*6*6, 320)
        # An affine transformation with 120 output features
        self.fc2 = nn.Linear(320, 120)
        # An affine transformation with number of classes output features
        self.fc3 = nn.Linear(120, 4)

        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x, T=1.0):
        # input should be of shape [b, c, w, h]
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        # print("ori: ", x.shape)

        # images 28x28 =>
        #      x.shape = [Batch_size, 1, 28, 28]
        x = F.relu(self.conv1(x))
        # Convolution with 3x3 filter, stride of 1, padding of 1 and 8 channels =>
        #     x.shape = [Batch_size, 8, 28, 28], since 28 - 3 + 2*padding + 1 = 28
        # Convolution with 3x3 filter, stride of 2, padding of 1 and 8 channels =>
        #     x.shape = [Batch_size, 8, 28, 28], since (28 - 3 + 2*padding)/2 + 1 = 14
        # print("relu1: ", x.shape)
        if not self.no_maxpool:
            x = self.maxPool1(x)
            # Max pooling with stride of 2 =>
            #     x.shape = [Batch_size, 8, 14, 14], since (28 - 2)/2 + 1 = 14
            # print("pool1: ", x.shape)
        
        x = F.relu(self.conv2(x))
        # Convolution with 3x3 filter, stride of 1, padding of 0 and 16 channels =>
        #     x.shape = [Batch_size, 16, 12, 12], since 12 = 14 - 3 + 1 >> maxpool
        #     x.shape = [Batch_size, 16, 6, 6], since 6 = (14 - 3)/2 + 1 >> no_maxpool
        # print("relu2: ", x.shape)
        if not self.no_maxpool:
            x = self.maxPool2(x)
            # Max pooling with stride of 2 =>
            #     x.shape = [Batch_size, 16, 6, 6], since (12 - 2)/2 + 1 = 6
            # print("pool2: ", x.shape)
        
        # print("flatten: ", x.shape)

        x = x.view(-1, 16*6*6)
        # print("view: ", x.shape)

        # prep for fully connected layer + relu
        x = F.relu(self.fc1(x))
        # print("relu3: ", x.shape)

        # drop out
        x = self.dropout(x)

        # second fully connected layer + relu
        x = F.relu(self.fc2(x))
        # print("relu4: ", x.shape)

        # last fully connected layer
        x = self.fc3(x)
        # print("fc3: ", x.shape)
        #exit(0)
        return F.log_softmax(x, dim=1)

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """

    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig(f"{IMAGE_PATH}/{IMAGE_NAME}_{name}.png", bbox_inches = 'tight')


def get_number_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def main():
    global IMAGE_PATH, IMAGE_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-image_path', type=str, default=IMAGE_PATH,
                        help="""The path which you want to save the generated plot""")
    parser.add_argument('-image_name', type=str, default=IMAGE_NAME,
                        help="""The name which you want to name the generated image""")
    parser.add_argument('-no_maxpool', action='store_true')

    opt = parser.parse_args()

    IMAGE_PATH = opt.image_path
    IMAGE_NAME = opt.image_name

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y
    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
