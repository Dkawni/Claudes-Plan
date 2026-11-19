from local_code.stage_4_code.Dataset_Loader import Dataset_Loader_Generation
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
        test_data_obj = Dataset_Loader_Jokes('data', '')
        test_data_obj.dataset_source_folder_path = '../../stage_4_data/text_generation/'

        result_obj = Result_Saver('saver', '')
        result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_Generation'
        result_obj.result_destination_file_name = 'Generated_Jokes'

    setting_obj = Setting_TrainingTest('????', '')


    # ---- running section ---------------------------------
    print('************ Start ************')

    test_data = test_data_obj.load()

    model = Method_RNN('rnn', '', vocab_size=len(train_data['dictionary']), rnn_type='lstm',
                       hidden_size=128, num_layers=1, bidirectional=False, dropout=0.0)
    model.data = {'train': {'X':train_data['X'], 'y' : train_data['Y']}, 'test' : {'X':test_data['X'], 'y' : test_data['Y']}}
    result = model.run()


    print('************ Overall Performance ************')
    for name, value in result.items():
        print(f'RNN {name}: {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------