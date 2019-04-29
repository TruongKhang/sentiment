import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Evaluation(object):
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true, dtype=np.int32)
        self.y_pred = np.array(y_pred, dtype=np.int32)
        metrics = precision_recall_fscore_support(y_true, y_pred, average='macro')
        self.precision_mac = metrics[0]
        self.recall_mac = metrics[1]
        self.f1_score_mac = metrics[2]
        self.accuracy = accuracy_score(y_true, y_pred)

if __name__ == '__main__':
    y_true = np.loadtxt('test_labels.txt')
    y_pred = np.loadtxt('predicted.txt')
    evaluater = Evaluation(y_true, y_pred)
    print(evaluater.precision_mac, evaluater.recall_mac, evaluater.f1_score_mac, evaluater.accuracy)