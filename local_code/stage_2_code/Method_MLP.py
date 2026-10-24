'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # which optimizer to use: 'adam' or 'sgd'
    optimizer_name = 'adam'
    # which loss function to use: 'ce' (cross entropy) or 'hinge' (multi-class hinge loss)
    loss_name = 'ce'

    # hidden_sizes lists the size of each hidden layer, e.g.
    # [128]     -> 784 -> 128 -> 10       (one hidden layer, the default)
    # [128, 64] -> 784 -> 128 -> 64 -> 10 (two hidden layers)
    def __init__(self, mName, mDescription, hidden_sizes=(128,)):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        layers = []
        in_size = 784
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))   # fully connected layer
            layers.append(nn.ReLU())               # activation function
            in_size = h
        layers.append(nn.Linear(in_size, 10))      # output layer: 10 class scores, no activation
        self.layers = nn.Sequential(*layers)
        # training loss of every epoch, used to draw the convergence plot
        self.loss_history = []

    def forward(self, x):
        '''Forward propagation'''
        # pass x through every layer in order; the output is an n x 10 score matrix
        return self.layers(x)

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_loss_function(self):
        if self.loss_name == 'hinge':
            return nn.MultiMarginLoss()
        # cross entropy applies softmax internally
        return nn.CrossEntropyLoss()

    def train(self, X, y):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_true = torch.LongTensor(np.array(y))
        self.loss_history = []

        # we don't do mini-batch, we use the whole input as one batch
        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_true)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # remember the loss of this epoch for the convergence plot
            self.loss_history.append(train_loss.item())

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        # instances get the label with the largest score
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)))
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        # re-initialize the weights of every linear layer
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}