'''
Stage 2 helper: load and summarize saved prediction results.

Reads the pickles written by Result_Saver into result/stage_2_result/ and
prints a short summary of each. Train/test runs are saved under the tag
'test'; k-fold runs are saved as '1', '2', '3', one per fold.

Run from anywhere:
    python3 script/stage_2_script/script_load_result.py
    python3 script/stage_2_script/script_load_result.py --prefix MLP_
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

from local_code.stage_2_code.Result_Loader import Result_Loader

RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_2_result')

# 'test' is written by Setting_TrainingTest; 1-3 by Setting_KFold_CV
FOLD_TAGS = ['test', 1, 2, 3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect saved stage 2 prediction results.')
    parser.add_argument(
        '--prefix', default='MLP_',
        help="result file prefix, e.g. 'MLP_', 'DT_', 'SVM_' (default: MLP_)")
    args = parser.parse_args()

    result_obj = Result_Loader('loader', '')
    result_obj.result_destination_folder_path = os.path.join(
        RESULT_DIR, args.prefix)
    result_obj.result_destination_file_name = 'prediction_result'

    found_any = False
    for fold_tag in FOLD_TAGS:
        path = '{}{}_{}'.format(
            result_obj.result_destination_folder_path,
            result_obj.result_destination_file_name,
            fold_tag)
        if not os.path.exists(path):
            continue

        found_any = True
        result_obj.fold_count = fold_tag
        result_obj.load()

        # summarize instead of dumping thousands of raw predictions
        pred_y = result_obj.data['pred_y']
        true_y = result_obj.data['true_y']
        correct = sum(int(p) == int(t) for p, t in zip(pred_y, true_y))
        print('Fold: {}, instances: {}, correct: {}, accuracy: {:.4f}'.format(
            fold_tag, len(true_y), correct, correct / len(true_y)))
        print('  first 10 predicted:', [int(v) for v in pred_y[:10]])
        print('  first 10 true     :', [int(v) for v in true_y[:10]])

    if not found_any:
        sys.exit(
            'No saved results found for prefix {!r} in {}\n'
            'Run a stage 2 script first, e.g. '
            'python3 script/stage_2_script/script_mlp.py'.format(
                args.prefix, RESULT_DIR))
