'''
Run the ablation experiments (the base model is run by script_mlp.py) (train on train.csv, test on test.csv),
then save the convergence plot and print a results table.
'''
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
import os
import numpy as np
import torch
import matplotlib
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..') + os.sep
matplotlib.use('Agg')  # save plots to files instead of opening windows
import matplotlib.pyplot as plt

# ---- experiments: each one changes ONE thing compared to the base model ----
experiments = [
    # name,                        hidden layers, loss,    optimizer, lr,   epochs
    ('More layers (128-64)',       [128, 64],     'ce',    'adam',    1e-3, 500),
    ('Wider layer (256)',          [256],         'ce',    'adam',    1e-3, 500),
    ('Hinge loss',                 [128],         'hinge', 'adam',    1e-3, 500),
    ('SGD optimizer (lr=0.1)',     [128],         'ce',    'sgd',     0.1,  500),
    ('1000 epochs',                [128],         'ce',    'adam',    1e-3, 1000),
]

result_folder = '../../result/stage_2_result/'
os.makedirs(result_folder, exist_ok=True)
all_scores = {}
all_losses = {}

for i, (name, hidden, loss_name, opt_name, lr, epochs) in enumerate(experiments):
    print('\n************ Experiment', i, ':', name, '************')
    # same random seed for every experiment so the comparison is fair
    np.random.seed(2)
    torch.manual_seed(2)

    train_data_obj = Dataset_Loader('train', '')
    train_data_obj.dataset_source_folder_path = ROOT + 'data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'

    test_data_obj = Dataset_Loader('test', '')
    test_data_obj.dataset_source_folder_path = ROOT + 'data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', name, hidden_sizes=hidden)
    method_obj.loss_name = loss_name
    method_obj.optimizer_name = opt_name
    method_obj.learning_rate = lr
    method_obj.max_epoch = epochs

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder + 'MLP_ablation' + str(i) + '_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_TrainingTest('train on train.csv, test on test.csv', '')
    setting_obj.test_dataset = test_data_obj

    setting_obj.prepare(train_data_obj, method_obj, result_obj, Evaluate_Accuracy('accuracy', ''))
    scores = setting_obj.load_run_save_evaluate()

    all_scores[name] = scores
    all_losses[name] = method_obj.loss_history
    for metric, value in scores.items():
        print(f'{name} {metric}: {value:.4f}')

# ---- plot: all ablation runs (for report section 3.7) ----
plt.figure(figsize=(7, 4.5))
for name, losses in all_losses.items():
    plt.plot(losses, label=name)
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('Training convergence of ablation experiments')
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(result_folder + 'convergence_ablation.png', dpi=200)
plt.close()

# ---- results table ----
print('\n************ Results on test.csv ************')
print(f'{"Experiment":30s} {"Accuracy":>9s} {"Precision":>10s} {"Recall":>8s} {"F1":>8s}')
for name, s in all_scores.items():
    print(f'{name:30s} {s["accuracy"]:9.4f} {s["precision"]:10.4f} {s["recall"]:8.4f} {s["f1"]:8.4f}')
print('\nPlots saved to', result_folder)