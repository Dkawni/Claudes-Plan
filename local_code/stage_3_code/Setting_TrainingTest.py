'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy, Evaluate_F, Evaluate_Recall, Evaluate_Precision
import os

class Setting_TrainingTest(setting):
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        # run
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw Result Module
        os.makedirs(os.path.dirname(self.result.result_destination_folder_path), exist_ok=True)
        self.result.data = learned_result
        self.result.fold_count = 'test'
        self.result.save()

        # evaluate on the testing set
        evaluators = {
            'accuracy': Evaluate_Accuracy('accuracy', ''),
            'precision': Evaluate_Precision('precision', ''),
            'recall': Evaluate_Recall('recall', ''),
            'f1': Evaluate_F('f1', ''),
        }
        scores = {}
        for name, evaluator in evaluators.items():
            evaluator.data = learned_result
            scores[name] = evaluator.evaluate()
        return scores