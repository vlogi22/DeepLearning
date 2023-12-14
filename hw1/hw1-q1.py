#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils

IMAGE_PATH = "./images"
IMAGE_NAME = "new_image"

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (1 x n_features): a single training example
        y_i (integer): the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Get probability scores according to the model 
        #   (num_labels x 1).
        label_scores = np.expand_dims(self.W.dot(x_i), axis = 1)

        # One-hot encode true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        # Softmax function
        # This gives the label probabilities according to the model 
        #   (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        
        # SGD update. W is (num_labels x num_features).
        # NOTE: [[1, 2, 3]] is a (1 x 3) matrix BUT [1, 2, 3] is only a vector
        #   dot() function have different behavior between them.
        self.W += learning_rate * (y_one_hot - label_probabilities).dot(np.expand_dims(x_i, axis = 1).T)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        units = [n_features, hidden_size, n_classes]

        # NOTE: input is  not included
        self.layers = 2
        self.g = [lambda x: x, lambda x: np.maximum(0, x), lambda x: x] # activation functions
        self.deriv_g = [np.vectorize(lambda x: 1), np.vectorize(lambda x: x > 0), np.vectorize(lambda x: 1)] # derivate of activation functions

        mu, sigma = 0.1, 0.1 # mean and standard deviation
        self.W = ["empty"] + [np.random.normal(mu, sigma, size = (b, a)) 
                    for a, b in zip(units[:-1], units[1:])]
        self.b = ["empty"] + [np.zeros(a) for a in units[1:]]

    def predict(self, X):
        """
        X (n_examples x n_features)
        """
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        prediction = []
        for x in X:
            output, _ = self.forward(x)
            prediction.append(np.argmax(output))

        return prediction

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        X (n_examples x n_features)
        """
        total_loss = np.array([])
        for x_i, y_i in zip(X, y):
            loss = self.update_weight(x_i, y_i, learning_rate)
            total_loss = np.append(total_loss, loss)

        return total_loss.mean()
    
    def update_weight(self, x, y, eta):
        y_one_hot = np.zeros((np.size(self.W[self.layers], 0), 1))
        y_one_hot[y] = 1
        # Compute forward pass
        y_hat, h = self.forward(x)
        # Compute Loss and Update total loss
        loss = self.compute_loss(y_hat, y_one_hot)
        
        # Compute backpropagation
        grad_weights, grad_biases = self.backward(y_one_hot, y_hat, h)
        # Update weights
        for i in range(1, self.layers + 1):
            self.W[i] -= eta*grad_weights[i]
            self.b[i] -= eta*np.reshape(grad_biases[i], (1, np.size(grad_biases[i], 0)))[0]
        
        return loss

    def compute_loss(self, y_hat, y):
        """
        y_hat (n classes x 1): prediction
        y (n classes x 1): gold labels
        """
        # Cross Entropy Loss
        y_hat = y_hat - np.max(y_hat) #To fix overflow
        probs = np.exp(y_hat) / np.sum(np.exp(y_hat))
        loss = -np.dot(y.T, np.log(probs))
        return loss[0]

    def forward(self, x):
        # compute hidden layers
        h = [x]

        for i in range(1, self.layers + 1):
            z = self.W[i].dot(h[i-1]) + self.b[i]
            h.append(self.g[i](z))
        return z, h

    def backward(self, y, y_hat, h):
        """
        y_hat (1 x n classes): predictions
        y (n classes x 1): gold labels
        """
        grad_weights = []
        grad_biases = []
        grad_h = [i for i in range(self.layers + 1)]
        grad_z = [i for i in range(self.layers + 1)]

        y_hat = y_hat - np.max(y_hat) #To fix overflow
        y_hat = np.expand_dims(y_hat, axis = 1)
        softmax = np.exp(y_hat) / np.sum(np.exp(y_hat))
        grad_z[self.layers] = softmax - y

        for i in range(self.layers, 0, -1):
            grad_h[i-1] = np.dot(self.W[i].T, grad_z[i])
            if i < self.layers:
                grad_z[i] = grad_h[i]*self.deriv_g[i](np.expand_dims(h[i], axis = 1))
            grad_weights = [np.dot(grad_z[i],  np.expand_dims(h[i-1], axis = 0))] + grad_weights
            grad_biases = [grad_z[i]] + grad_biases
        
        grad_weights = ["empty"] + grad_weights
        grad_biases = ["empty"] + grad_biases
        return grad_weights, grad_biases


def plot(epochs, train_accs, val_accs):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig(f"{IMAGE_PATH}/{IMAGE_NAME}_accuracy.png", bbox_inches = 'tight')

def plot_loss(epochs, loss):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig(f"{IMAGE_PATH}/{IMAGE_NAME}_loss.png", bbox_inches = 'tight')


def main():
    global IMAGE_PATH, IMAGE_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-image_path', type=str, default=IMAGE_PATH,
                        help="""The path which you want to save the generated plot""")
    parser.add_argument('-image_name', type=str, default=IMAGE_NAME,
                        help="""The name which you want to name the generated image""")
    opt = parser.parse_args()

    IMAGE_PATH = opt.image_path
    IMAGE_NAME = opt.image_name

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
