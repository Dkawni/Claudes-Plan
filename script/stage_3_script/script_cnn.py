from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN as Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN as Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN as Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy, Evaluate_Precision, Evaluate_Recall, Evaluate_F
from local_code.stage_3_code.Plot_Convergence import plot_training_convergence
from local_code.stage_3_code.Setting_TrainingTest import Setting_TrainingTest
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    if 1:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'MNIST'

        method_obj = Method_CNN_MNIST('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
        result_obj.result_destination_file_name = 'prediction_result_MNIST'

    if 0:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'ORL'

        method_obj = Method_CNN_ORL('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
        result_obj.result_destination_file_name = 'prediction_result_ORL'

    if 0:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'CIFAR'

        method_obj = Method_CNN_CIFAR('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
        result_obj.result_destination_file_name = 'prediction_result_CIFAR'

    method_obj.val_ratio = 0.0
    setting_obj = Setting_TrainingTest('train on train set, test on test set', '')
    # setting_obj = Setting_Train_Test_Split('train test split', '')

    accuracy_obj = Evaluate_Accuracy('accuracy', '')
    precision_obj = Evaluate_Precision('precision', '')
    recall_obj = Evaluate_Recall('recall', '')
    f1_obj = Evaluate_F('f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(test_data_obj, method_obj, result_obj, accuracy_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate()
    plot_training_convergence(method_obj.loss_history, None, result_obj.result_destination_folder_path + 'training_convergence.png', title='CNN Training Convergence (' + test_data_obj.dataset_source_file_name + ')')
    print('************ Overall Performance ************')
    for name, value in result.items():
        print(f'CNN {name}: {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------