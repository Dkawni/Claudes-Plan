from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN as Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN as Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN as Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ---- configuration changes ------------------------

def fix_fc1(m, input_shape):
    # recompute the input size of fc1 after the conv layers change
    with torch.no_grad():
        x = torch.zeros(1, *input_shape)
        x = m.pool(F.relu(m.conv1(x)))
        x = m.pool(F.relu(m.conv2(x)))
    m.fc1 = nn.Linear(x.numel(), m.fc1.out_features)


def baseline(m, s):
    pass

def kernel_5(m, s):
    m.conv1 = nn.Conv2d(m.conv1.in_channels, m.conv1.out_channels, kernel_size=5, padding=2)
    m.conv2 = nn.Conv2d(m.conv2.in_channels, m.conv2.out_channels, kernel_size=5, padding=2)

def avg_pooling(m, s):
    m.pool = nn.AvgPool2d(2, 2)

def deeper(m, s):
    c = m.conv1.out_channels
    m.conv2 = nn.Sequential(nn.Conv2d(c, c, kernel_size=3, padding=1), nn.ReLU(), m.conv2)

configs = [baseline, kernel_5, deeper, avg_pooling]

# ----------------------------------

if 1:
    dataset_name = 'MNIST'   # 'MNIST', 'ORL' or 'CIFAR'
    ablation_epochs = None

# --------------------------------
input_shape = {'MNIST': (1, 28, 28), 'ORL': (1, 112, 92), 'CIFAR': (3, 32, 32)}[dataset_name]
result_folder = '../../result/stage_3_result/ablation_' + dataset_name + '/'
os.makedirs(result_folder, exist_ok=True)

all_scores = {}
all_losses = {}
for config in configs:
    print('************ Config:', config.__name__, '************')
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader(dataset_name, '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = dataset_name

    if dataset_name == 'MNIST':
        method_obj = Method_CNN_MNIST('convolutional neural network', '')
    elif dataset_name == 'ORL':
        method_obj = Method_CNN_ORL('convolutional neural network', '')
    else:
        method_obj = Method_CNN_CIFAR('convolutional neural network', '')
    if ablation_epochs is not None:
        method_obj.max_epoch = ablation_epochs
    config(method_obj, input_shape)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = result_folder
    result_obj.result_destination_file_name = config.__name__

    setting_obj = Setting_TrainingTest('train on train set, test on test set', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    scores = setting_obj.load_run_save_evaluate()
    all_scores[config.__name__] = scores
    all_losses[config.__name__] = method_obj.loss_history

# loss curves-----------------------
plt.figure(figsize=(8, 5))
for name, losses in all_losses.items():
    plt.plot(range(len(losses)), losses, label=name)
plt.xlabel('Training epoch')
plt.ylabel('Training loss')
plt.title('CNN Ablation Study (' + dataset_name + ')')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(result_folder + 'ablation_loss_curves.png', dpi=150)
print('saved to', result_folder)