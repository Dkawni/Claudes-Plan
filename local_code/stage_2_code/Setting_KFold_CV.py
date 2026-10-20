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
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)

        evaluators = {
            'accuracy': Evaluate_Accuracy('accuracy', ''),
            'precision': Evaluate_Precision('precision', ''),
            'recall': Evaluate_Recall('recall', ''),
            'f1': Evaluate_F('f1', ''),
        }
        scores = {name: [] for name in evaluators}

        fold_count = 0
        for train_index, test_index in kf.split(loaded_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = np.array(loaded_data['X'])[train_index], np.array(loaded_data['X'])[test_index]
            y_train, y_test = np.array(loaded_data['y'])[train_index], np.array(loaded_data['y'])[test_index]
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

        return {name: (np.mean(v), np.std(v)) for name, v in scores.items()}

        