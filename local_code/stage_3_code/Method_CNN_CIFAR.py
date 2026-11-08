from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy
from local_code.base_class.method import method
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 15
    learning_rate = 1e-3
    optimizer_name = 'adam'
    loss_name = 'ce'
    val_ratio = 0.15

    def __init__(self, mName, mDescription):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)
        self.dropout = nn.Dropout(0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # (N,3,32,32)  -> (N,64,32,32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # (N,64,16,16) -> (N,128,16,16)
        self.pool = nn.MaxPool2d(2, 2)  # halves spatial each use
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # 8192 features in
        self.fc2 = nn.Linear(128, 10)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(128)


        # ---------- data ----------
    def _make_dataset(self, split):
        d = self.data[split]
        X = torch.as_tensor(np.asarray(d['X']), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(d['y']), dtype=torch.long)
        # MNIST: (N, 784) -> (N, 1, 28, 28)
        if X.dim() == 3 and X.shape[1] == 32 * 32 * 3:
            X = X.view(-1, 3, 32, 32)
        return TensorDataset(X, y)

    def split(self):
        full = self._make_dataset('train')
        val_size = int(len(full) * self.val_ratio)
        train_size = len(full) - val_size
        train_subset, val_subset = random_split(full, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        return train_loader, val_loader

    # ---------- model ----------
    def forward(self, x):
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))  # (N,32,56,46)
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x)))) # (N,64,28,23)

        # dropout implemented
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

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
        train_loader, val_loader = self.split()

        self.loss_history = []
        self.val_loss_history = []
        self.val_accuracy_history = []

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
            self.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    y_pred = self.forward(inputs)
                    val_loss_sum += loss_function(y_pred, targets).item() * len(targets)
                    val_correct += (y_pred.max(1)[1] == targets).sum().item()
                    val_total += len(targets)
            self.train()
            if val_total > 0:
                self.val_loss_history.append(val_loss_sum / val_total)
                self.val_accuracy_history.append(val_correct / val_total)
            if epoch % 5 == 0:
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
            if X.dim() == 3 and X.shape[1] == 32 * 32 * 3:
                X = X.view(-1, 3, 32, 32)
            y_pred = self.forward(X)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}