from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])


class Evaluate_Precision(evaluate):
    data = None

    def evaluate(self):
        return precision_score(self.data['true_y'], self.data['pred_y'],
                               average='macro', zero_division=0)


class Evaluate_Recall(evaluate):
    data = None

    def evaluate(self):
        return recall_score(self.data['true_y'], self.data['pred_y'],
                            average='macro', zero_division=0)


class Evaluate_F(evaluate):
    data = None

    def evaluate(self):
        return f1_score(self.data['true_y'], self.data['pred_y'],
                        average='macro', zero_division=0)