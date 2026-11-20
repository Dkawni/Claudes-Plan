from local_code.stage_3_code.Result_Loader import Result_Loader
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Accuracy, Evaluate_Precision, Evaluate_Recall, Evaluate_F
if 1:
    for dataset_name in ['MNIST', 'ORL', 'CIFAR']:
        result_obj = Result_Loader('loader', '')
        result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_' + dataset_name + '_'
        result_obj.result_destination_file_name = 'prediction_result_' + dataset_name
        result_obj.fold_count = 'test'
        result_obj.load()

        evaluators = {
            'accuracy': Evaluate_Accuracy('accuracy', ''),
            'precision': Evaluate_Precision('precision', ''),
            'recall': Evaluate_Recall('recall', ''),
            'f1': Evaluate_F('f1', ''),
        }
        print('************', dataset_name, '************')
        for name, evaluator in evaluators.items():
            evaluator.data = result_obj.data
            print(f'CNN {name}: {evaluator.evaluate():.4f}')


# ---- ablation results ----
if 1:
    for dataset_name in ['MNIST']:
        print('************ Ablation results (' + dataset_name + ') ************')
        print(f"{'config':<15}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}")
        for config_name in ['baseline', 'kernel_5', 'deeper', 'avg_pooling']:
            result_obj = Result_Loader('loader', '')
            result_obj.result_destination_folder_path = '../../result/stage_3_result/ablation_' + dataset_name + '/'
            result_obj.result_destination_file_name = config_name
            result_obj.fold_count = 'test'
            result_obj.load()

            scores = []
            for evaluator in [Evaluate_Accuracy('accuracy', ''), Evaluate_Precision('precision', ''),
                              Evaluate_Recall('recall', ''), Evaluate_F('f1', '')]:
                evaluator.data = result_obj.data
                scores.append(evaluator.evaluate())
            print(f"{config_name:<15}{scores[0]:>10.4f}{scores[1]:>11.4f}{scores[2]:>9.4f}{scores[3]:>9.4f}")