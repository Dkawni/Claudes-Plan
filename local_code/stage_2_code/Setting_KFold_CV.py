'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from local_code.stage_2_code.Evaluate_Metrics import (
    Evaluate_Accuracy, Evaluate_Precision, Evaluate_Recall, Evaluate_F)
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    fold = 3
    random_state = 2

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        # final training loss of each fold, for methods that expose a
        # loss_history (e.g. Method_MLP). Empty for DT / SVM.
        self.fold_losses = []
        # full per-epoch loss curve of each fold, used for convergence plots
        self.fold_loss_histories = []

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        # seed the shuffle so the reported mean/std are reproducible
        kf = KFold(n_splits=self.fold, shuffle=True,
                   random_state=self.random_state)

        evaluators = {
            'accuracy': Evaluate_Accuracy('accuracy', ''),
            'precision': Evaluate_Precision('precision', ''),
            'recall': Evaluate_Recall('recall', ''),
            'f1': Evaluate_F('f1', ''),
        }
        scores = {name: [] for name in evaluators}

        self.fold_losses = []
        self.fold_loss_histories = []

        X_all = np.array(loaded_data['X'])
        y_all = np.array(loaded_data['y'])

        fold_count = 0
        for train_index, test_index in kf.split(X_all):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = X_all[train_index], X_all[test_index]
            y_train, y_test = y_all[train_index], y_all[test_index]
            # run MethodModule
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()

            # save raw ResultModule
            self.result.data = learned_result
            self.result.fold_count = fold_count
            self.result.save()

            for name, ev in evaluators.items():
                ev.data = learned_result
                scores[name].append(ev.evaluate())

            # record the training loss for methods that track it. Gradient-based
            # methods (Method_MLP) expose loss_history; DT and SVM do not.
            loss_history = getattr(self.method, 'loss_history', None)
            if loss_history:
                self.fold_losses.append(loss_history[-1])
                self.fold_loss_histories.append(list(loss_history))

        results = {name: (np.mean(v), np.std(v)) for name, v in scores.items()}

        # report loss last, after the four classification metrics
        if self.fold_losses:
            results['loss'] = (np.mean(self.fold_losses),
                               np.std(self.fold_losses))

        return results
