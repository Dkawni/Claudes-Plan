from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN as Method_CNN_MNIST
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN as Method_CNN_ORL
from local_code.stage_3_code.Method_CNN_CIFAR import Method_CNN as Method_CNN_CIFAR
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy, Evaluate_Precision, Evaluate_Recall, Evaluate_F
from local_code.stage_3_code.Plot_Convergence import plot_training_convergence
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    if 0:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'MNIST'

        method_obj = Method_CNN_MNIST('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_MNIST_'
        result_obj.result_destination_file_name = 'prediction_result_MNIST'

    if 0:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../local_code/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'ORL'

        method_obj = Method_CNN_ORL('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_ORL_'
        result_obj.result_destination_file_name = 'prediction_result_ORL'

    if 1:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'CIFAR'

        method_obj = Method_CNN_CIFAR('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
        result_obj.result_destination_file_name = 'prediction_result_CIFAR'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    # setting_obj = Setting_Train_Test_Split('train test split', '')

    accuracy_obj = Evaluate_Accuracy('accuracy', '')
    precision_obj = Evaluate_Precision('precision', '')
    recall_obj = Evaluate_Recall('recall', '')
    f1_obj = Evaluate_F('f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    data = test_data_obj.load()
    method_obj.data = data
    output = method_obj.run()
    accuracy_obj.data = output
    precision_obj.data = output
    recall_obj.data = output
    f1_obj.data = output
    plot_training_convergence(method_obj.loss_history, None, '../../result/stage_3_result/CNN_MNIST_training_convergence.png')
    result = {'accuracy': accuracy_obj.evaluate(), 'precision': precision_obj.evaluate(), 'recall': recall_obj.evaluate(), 'f1': f1_obj.evaluate()}
    print('************ Overall Performance ************')
    for name, value in result.items():
        print(f'CNN {name}: {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------