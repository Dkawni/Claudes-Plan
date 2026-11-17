from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Evaluate_Metrics import Evaluate_Accuracy, Evaluate_Precision, Evaluate_Recall, Evaluate_F
from local_code.stage_4_code.Plot_Convergence import plot_training_convergence
from local_code.stage_4_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_4_code.Method_RNN_CLASSIFICATION import Method_RNN
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
        test_data_obj = Dataset_Loader('test', '')
        test_data_obj.dataset_source_folder_path = '../../stage_4_data/text_classification/test/'

        train_data_obj = Dataset_Loader('train', '')
        train_data_obj.dataset_source_folder_path = '../../stage_4_data/text_classification/train/'

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN'
        result_obj.result_destination_file_name = 'prediction_result_ORL'

    if 0:
        test_data_obj = Dataset_Loader('data', '')
        test_data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
        test_data_obj.dataset_source_file_name = 'CIFAR'

        model = Method_CNN_CIFAR('convolution neural network', '')

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_CIFAR_'
        result_obj.result_destination_file_name = 'prediction_result_CIFAR'
    setting_obj = Setting_TrainingTest('????', '')


    accuracy_obj = Evaluate_Accuracy('accuracy', '')
    precision_obj = Evaluate_Precision('precision', '')
    recall_obj = Evaluate_Recall('recall', '')
    f1_obj = Evaluate_F('f1', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    test_data = test_data_obj.load()
    train_data = train_data_obj.load()

    model = Method_RNN('rnn', '', vocab_size=len(train_data['dictionary']), rnn_type='lstm',
                       hidden_size=128, num_layers=1, bidirectional=False, dropout=0.0)
    model.data = {'train': {'X':train_data['X'], 'y' : train_data['Y']}, 'test' : {'X':test_data['X'], 'y' : test_data['Y']}}
    result = model.run()

    setting_obj.prepare(test_data_obj, model, result_obj, accuracy_obj)
    setting_obj.print_setup_summary()
    result = setting_obj.load_run_save_evaluate()
    plot_training_convergence(model.loss_history, None,
                             result_obj.result_destination_folder_path + 'training_convergence.png',
                             title='RNN Training Convergence (' + test_data_obj.dataset_source_file_name + ')',
                             val_loss_history = model.val_loss_history)
    print('************ Overall Performance ************')
    for name, value in result.items():
        print(f'RNN {name}: {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------