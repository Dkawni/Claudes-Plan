'''
Stage 2: Multi-Layer Perceptron on the MNIST handwritten-digit dataset.

Two evaluation protocols are available:

  default   train on data/stage_2_data/train.csv (60,000 images) and evaluate
            on data/stage_2_data/test.csv (10,000 images)
  --kfold   3-fold cross-validation on train.csv, reporting mean +/- std

Both report Accuracy, Precision, Recall, F1 and the final training loss, and
write a training-convergence plot.

Run from anywhere:
    python3 script/stage_2_script/script_mlp.py
    python3 script/stage_2_script/script_mlp.py --kfold
'''

import argparse
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

from local_code.stage_2_code.Dataset_Subsample import Dataset_Subsample
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
from local_code.stage_2_code.Plot_Convergence import (
    plot_training_convergence, plot_kfold_convergence)

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')


def print_scores(label, scores):
    '''
    Print a score dict. Setting_TrainingTest returns {name: value} while
    Setting_KFold_CV returns {name: (mean, std)}, so handle both shapes.
    '''
    for name, value in scores.items():
        if isinstance(value, (tuple, list)):
            print('{} {}: {:.4f} +/- {:.4f}'.format(label, name, value[0], value[1]))
        else:
            print('{} {}: {:.4f}'.format(label, name, value))


# ---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MLP on stage 2 MNIST data.')
    parser.add_argument(
        '--kfold', action='store_true',
        help='run 3-fold cross-validation on train.csv and report mean +/- std, '
             'instead of training on train.csv and testing on test.csv')
    parser.add_argument(
        '--folds', type=int, default=3, metavar='K',
        help='number of cross-validation folds when --kfold is used (default: 3)')
    parser.add_argument(
        '--fold-seed', type=int, default=2, metavar='S',
        help='seed for the k-fold shuffle (default: 2). Changing it reshuffles '
             'the fold boundaries, which moves the scores slightly.')
    parser.add_argument(
        '--subset', type=int, default=None, metavar='N',
        help='train on a stratified subsample of N images (default: all)')
    parser.add_argument(
        '--epochs', type=int, default=None, metavar='E',
        help='training epochs (default: the model default, 500)')
    args = parser.parse_args()

    # ---- parameter section -------------------------------
    # fixed seeds so the reported numbers are reproducible
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    os.makedirs(RESULT_DIR, exist_ok=True)

    needed = ['train.csv'] if args.kfold else ['train.csv', 'test.csv']
    for split in needed:
        if not os.path.exists(os.path.join(DATA_DIR, split)):
            sys.exit(
                'Missing dataset file: {}\n'
                'Place the stage 2 MNIST CSVs in {} before running. '
                'See README.md for details.'.format(
                    os.path.join(DATA_DIR, split), DATA_DIR))

    # ---- object initialization section -------------------
    train_data_obj = Dataset_Subsample('train', '')
    train_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    train_data_obj.dataset_source_file_name = 'train.csv'
    train_data_obj.max_samples = args.subset

    test_data_obj = Dataset_Subsample('test', '')
    test_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')
    if args.epochs:
        method_obj.max_epoch = args.epochs

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join(RESULT_DIR, 'MLP_')
    result_obj.result_destination_file_name = 'prediction_result'

    if args.kfold:
        # k-fold splits a single dataset internally, so it only takes train.csv
        setting_obj = Setting_KFold_CV('k fold cross validation', '')
        setting_obj.fold = args.folds
        setting_obj.random_state = args.fold_seed
    else:
        setting_obj = Setting_TrainingTest(
            'train on train.csv, test on test.csv', '')
        setting_obj.test_dataset = test_data_obj

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    results = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print_scores('MLP', results)

    # ---- training convergence plot (report section 3.5) --
    if args.kfold:
        plot_kfold_convergence(
            fold_loss_histories=setting_obj.fold_loss_histories,
            save_path=os.path.join(
                RESULT_DIR, 'mlp_training_convergence_kfold.png'),
            title='MLP Training Convergence per Fold ({} epochs)'.format(
                method_obj.max_epoch))
    else:
        plot_training_convergence(
            loss_history=method_obj.loss_history,
            accuracy_history=method_obj.accuracy_history,
            save_path=os.path.join(RESULT_DIR, 'mlp_training_convergence.png'),
            title='MLP Training Convergence on MNIST ({} epochs)'.format(
                method_obj.max_epoch))

    print('************ Finish ************')
    # ------------------------------------------------------
