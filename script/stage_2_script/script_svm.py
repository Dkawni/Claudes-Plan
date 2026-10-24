'''
Stage 2 baseline: Support Vector Machine on the MNIST handwritten-digit dataset.

Uses the same protocol as script_mlp.py -- train on train.csv, evaluate on
test.csv -- so the scores are directly comparable to the MLP's.

Kernel SVC scales roughly quadratically with the training set size, so fitting
all 60,000 images takes hours. Pass --subset to train on a stratified sample:

    python3 script/stage_2_script/script_svm.py --subset 10000
    python3 script/stage_2_script/script_svm.py              # full 60,000
    python3 script/stage_2_script/script_svm.py --kfold --subset 10000
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

from local_code.stage_2_code.Dataset_Subsample import Dataset_Subsample
from local_code.stage_2_code.Method_SVM import Method_SVM
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy

DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_2_data')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')

# training-set size above which we warn that the fit will be slow
SLOW_FIT_THRESHOLD = 20000


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


# ---- Support Vector Machine script ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SVM baseline on stage 2 MNIST data.')
    parser.add_argument(
        '--kfold', action='store_true',
        help='run 3-fold cross-validation on train.csv instead of '
             'training on train.csv and testing on test.csv')
    parser.add_argument(
        '--subset', type=int, default=None, metavar='N',
        help='train on a stratified subsample of N images (default: all). '
             'Strongly recommended -- a full fit takes hours.')
    parser.add_argument(
        '-C', type=float, default=1.0,
        help='SVC regularization parameter C (default: 1.0)')
    args = parser.parse_args()

    # ---- parameter section -------------------------------
    c = args.C
    # fixed seed so the reported numbers are reproducible
    np.random.seed(2)
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

    if args.subset is None or args.subset > SLOW_FIT_THRESHOLD:
        print('WARNING: fitting SVC on more than {} images is very slow '
              '(hours). Consider --subset 10000.'.format(SLOW_FIT_THRESHOLD))

    # ---- object initialization section -------------------
    # NOTE: train and test each need their own loader object. Assigning both
    # to one variable (as an earlier version of this script did) throws the
    # first loader away and silently evaluates on the wrong split.
    train_data_obj = Dataset_Subsample('train', '')
    train_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    train_data_obj.dataset_source_file_name = 'train.csv'
    train_data_obj.max_samples = args.subset

    test_data_obj = Dataset_Subsample('test', '')
    test_data_obj.dataset_source_folder_path = DATA_DIR + os.sep
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_SVM('support vector machine', '')
    method_obj.c = c

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = os.path.join(RESULT_DIR, 'SVM_')
    result_obj.result_destination_file_name = 'prediction_result'

    if args.kfold:
        # k-fold splits a single dataset internally, so it only takes train.csv
        setting_obj = Setting_KFold_CV('k fold cross validation', '')
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

    # NOTE: both settings return a dict of metrics. Unpacking the result into
    # two variables (as an earlier version did) raises a ValueError.
    scores = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print_scores('SVM', scores)
    print('************ Finish ************')
    # ------------------------------------------------------
