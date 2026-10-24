from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_TrainingTest import Setting_TrainingTest
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    test_data_obj = Dataset_Loader('test', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'

    train_data_obj = Dataset_Loader('train', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Train_Test_Split('train test split', '')
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
        print(f'MLP {name}: {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    