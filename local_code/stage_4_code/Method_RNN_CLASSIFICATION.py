from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Accuracy
from local_code.base_class.method import method
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import torch.nn as nn
import numpy as np

class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 3
    learning_rate = 1e-3
    optimizer_name = 'adam'
    loss_name = 'ce'
    val_ratio = 0.15

    def __init__(self, mName, mDescription, *,
                 vocab_size, embed_dim=128, hidden_size=128,
                 num_layers=1, num_classes=2, rnn_type='lstm',
                 bidirectional=False, dropout=0.0):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        rnn_cls = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    @staticmethod
    def collate_fn(batch):
        seqs, labels = zip(*batch)  # seqs: tuple of 1D long tensors
        lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
        padded = nn.utils.rnn.pad_sequence(
            list(seqs), batch_first=True, padding_value=0,
        )
        return padded, torch.tensor(labels, dtype=torch.long), lengths

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        packed_out, h_n = self.rnn(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # unpadded to max length
        # changed to mean pooling: average the output for all real words
        mask = (torch.arange(out.size(1))[None, :] < lengths[:, None]).unsqueeze(2).float()
        mean = (out * mask).sum(dim=1) / lengths[:, None].float()
        return self.fc(self.dropout(mean))

    def _make_dataset(self, split):
        d = self.data[split]
        return [(torch.tensor(seq, dtype=torch.long), int(label))
                for seq, label in zip(d['X'], d['y'])]

    def split(self):
        full = self._make_dataset('train')
        val_size = int(len(full) * self.val_ratio)
        train_size = len(full) - val_size
        train_subset, val_subset = random_split(full, [train_size, val_size])
        self.train_loader = DataLoader(train_subset, batch_size=64, shuffle=True,
                                       collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_subset, batch_size=64, shuffle=False,
                                     collate_fn=self.collate_fn)
        return self.train_loader, self.val_loader

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_loss_function(self):
        if self.loss_name == 'hinge':
            return nn.MultiMarginLoss()
        return nn.CrossEntropyLoss()

    def test(self, X):
        self.eval()
        with torch.no_grad():
            ds = [(torch.tensor(seq, dtype=torch.long), 0) for seq in X]
            loader = DataLoader(ds, batch_size=64, shuffle=False,
                                collate_fn=self.collate_fn)
            preds = []
            for inputs, _, lengths in loader:
                preds.append(self.forward(inputs, lengths).argmax(dim=1))
        return torch.cat(preds)

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

            for inputs, targets, lengths in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(inputs, lengths)
                loss = loss_function(y_pred, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
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
                for inputs, targets, lengths in val_loader:
                    y_pred = self.forward(inputs, lengths)
                    val_loss_sum += loss_function(y_pred, targets).item() * len(targets)
                    val_correct += (y_pred.max(1)[1] == targets).sum().item()
                    val_total += len(targets)
            self.train()
            if val_total > 0:
                self.val_loss_history.append(val_loss_sum / val_total)
                self.val_accuracy_history.append(val_correct / val_total)


            accuracy_evaluator.data = {
                'true_y': last_targets.cpu().numpy(),
                'pred_y': last_preds.cpu().numpy(),
            }
            print(f"Epoch: {epoch}  Accuracy: {accuracy_evaluator.evaluate()}  Loss: {avg_loss:.4f}")

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model()
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}