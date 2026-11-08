from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy
from local_code.base_class.method import method
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 10
    learning_rate = 1e-3
    optimizer_name = 'adam'
    loss_name = 'ce'
    val_ratio = 0.15

    def __init__(self, mName, mDescription, hidden_sizes=(128,)):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.conv1 = nn.Conv2d(1, 28, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(28, 14, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(14 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.loss_history = []
        self.train_loader = None
        self.val_loader = None

    # ---------- data ----------
    def _make_dataset(self, split):
        d = self.data[split]
        X = torch.as_tensor(np.asarray(d['X']), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(d['y']), dtype=torch.long)
        # MNIST: (N, 784) -> (N, 1, 28, 28)
        if X.dim() == 2 and X.shape[1] == 28 * 28:
            X = X.view(-1, 1, 28, 28)
        return TensorDataset(X, y)

    def split(self):
        full = self._make_dataset('train')
        val_size = int(len(full) * self.val_ratio)
        train_size = len(full) - val_size
        train_subset, val_subset = random_split(full, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        return self.train_loader, self.val_loader

    # ---------- model ----------
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (N, 1, 28, 28) -> (N, 28, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # -> (N, 14, 7, 7)
        x = torch.flatten(x, 1)                # -> (N, 686)
        x = F.relu(self.fc1(x))                # -> (N, 128)
        x = self.fc2(x)                        # -> (N, 10)
        return x

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_loss_function(self):
        if self.loss_name == 'hinge':
            return nn.MultiMarginLoss()
        return nn.CrossEntropyLoss()

    # ---------- training ----------
    def train_model(self):
        self.train()                             # nn.Module.train -> training mode
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        train_loader, _ = self.split()
        self.loss_history = []

        for epoch in range(self.max_epoch):
            running_loss = 0.0
            n_batches = 0
            last_preds = None
            last_targets = None

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(inputs)
                loss = loss_function(y_pred, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1
                last_preds = y_pred.detach().max(1)[1]
                last_targets = targets

            avg_loss = running_loss / max(n_batches, 1)
            self.loss_history.append(avg_loss)


            accuracy_evaluator.data = {
                'true_y': last_targets.cpu().numpy(),
                'pred_y': last_preds.cpu().numpy(),
            }
            print(f"Epoch: {epoch}  Accuracy: {accuracy_evaluator.evaluate()}  Loss: {avg_loss:.4f}")

    # ---------- testing ----------
    def test(self, X):
        self.eval()                              # eval mode
        with torch.no_grad():
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            if X.dim() == 2 and X.shape[1] == 28 * 28:
                X = X.view(-1, 1, 28, 28)
            y_pred = self.forward(X)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}