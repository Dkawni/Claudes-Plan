"""
Concrete SettingModule class for training the given testing set
"""

from local_code.base_class.setting import setting
from local_code.stage_2_code.Evaluate_Metrics import (
    Evaluate_Accuracy, Evaluate_Recall, Evaluate_F, Evaluate_Precision)
import numpy as np

class Setting_TrainingTest(setting):
    test_dataset = None

    def load_run_save_evaluate(self):
    # load the training and testing set separately
        train_data = self.dataset.load()
        test_data = self.test_dataset.load()

    # train on train.csv and predict on test.csv
        self.method.data = {
            'train': {'X': np.array(train_data['X']), 'y': np.array(train_data['y'])},
            'test': {'X': np.array(test_data['X']), 'y': np.array(test_data['y'])}
        }
        learned_result = self.method.run()

    # save the predictions
        self.result.data = learned_result
        self.result.fold_count = 'test'
        self.result.save()

    # score predictions
        evaluators = {
        'accuracy': Evaluate_Accuracy('accuracy', ''),
        'recall': Evaluate_Recall('recall', ''),
        'precision': Evaluate_Precision('precision', ''),
        'f1' : Evaluate_F('f1', ''),
        }

        scores = {}
        for name, ev in evaluators.items():
            ev.data = learned_result
            scores[name] = ev.evaluate()

        return scores