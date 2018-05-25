import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

y_true = np.loadtxt('test_labels.txt')
y_pred = np.loadtxt(sys.argv[1])
(precision_mac, recall_mac, fscore_mac, support_mac) = precision_recall_fscore_support(y_true, y_pred, average='macro')
accuracy = accuracy_score(y_true, y_pred)

print(precision_mac, recall_mac, fscore_mac, accuracy)
