import sys
from sklearn.svm import LinearSVC
import numpy as np

from dataset import represent_doc_embedding
from evaluation import Evaluation

def do_classification(data, C):
    print('Classification by SVM...')
    training_docs, training_labels, test_docs, test_labels = data[0], data[1], data[2], data[3]
    svm = LinearSVC(C=C, class_weight={0: 0.3, 1: 0.7})
    svm.fit(training_docs, training_labels)
    labels_pred = svm.predict(test_docs)

    evaluater = Evaluation(test_labels, labels_pred)
    print('Evaluation for predicting sara comment:')
    print('\tMacro-Precision: ', evaluater.precision_mac)
    print('\tMacro-Recall: ', evaluater.recall_mac)
    print('\tMacro-F1: ', evaluater.f1_score_mac)
    print('\tAccuracy: ', evaluater.accuracy)

if __name__ == '__main__':
    word_embedding_file = sys.argv[1]
    training_file = 'data/training_data_real_vocab.txt'
    test_file = 'data/test_data_real_vocab.txt'
    word_embedding = np.load(word_embedding_file)
    new_data = represent_doc_embedding(training_file, test_file, word_embedding=word_embedding, type='concat')
    do_classification(new_data, C=1.0)
