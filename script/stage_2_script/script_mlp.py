'''
Stage 2: Multi-Layer Perceptron on the MNIST handwritten-digit dataset.

Trains on data/stage_2_data/train.csv (60,000 images) and evaluates on
data/stage_2_data/test.csv (10,000 images), reporting Accuracy, Precision,
Recall and F1, and writing a training-convergence plot.

Run from anywhere:
    python3 script/stage_2_script/script_mlp.py
'''

import os
import sys

# Make the project importable and resolve data/result paths relative to the
# repository root, so this script works regardless of the current directory.
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
from local_code.stage_2_code.Plot_Convergence import plot_training_convergence

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')

# ---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    # ---- parameter section -------------------------------
    # fixed seeds so the reported numbers are reproducible
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    os.makedirs(RESULT_DIR, exist_ok=True)

    for split in ('train.csv', 'test.csv'):
        if not os.path.exists(os.path.join(DATA_DIR, split)):
            sys.exit(
                'Missing dataset file: {}\n'
                'Place the stage 2 MNIST CSVs in {} before running. '
                'See README.md for details.'.format(
                    os.path.join(DATA_DIR, split), DATA_DIR))

    # ---- object initialization section -------------------
    train_data_obj = Dataset_Loader('train', '')
    train_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    train_data_obj.dataset_source_file_name = 'train.csv'

    test_data_obj = Dataset_Loader('test', '')
    test_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join(RESULT_DIR, 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_TrainingTest('train on train.csv, test on test.csv', '')
    setting_obj.test_dataset = test_data_obj

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    results = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    for name, value in results.items():
        print('MLP {}: {:.4f}'.format(name, value))

    # ---- training convergence plot (report section 3.5) --
    plot_training_convergence(
        loss_history=method_obj.loss_history,
        accuracy_history=method_obj.accuracy_history,
        save_path=os.path.join(RESULT_DIR, 'mlp_training_convergence.png'),
        title='MLP Training Convergence on MNIST ({} epochs)'.format(
            method_obj.max_epoch))

    print('************ Finish ************')
    # ------------------------------------------------------
