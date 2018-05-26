import sys, os
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, subtract
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.svm import LinearSVC

from dataset import load_dataset, represent_doc_embedding
from evaluation import Evaluation

def model_sswe_h(window_size, vocab_size, embedding_size):
    input = Input(shape=(window_size,), dtype='int32', name='input')
    embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=window_size, name='embedding')(input)
    concat_layer = Flatten(name='concatnate')(embedding)
    hidden_layer1 = Dense(20, activation='tanh', name='hidden')(concat_layer)
    out_layer = Dense(2, activation='softmax', name='output')(hidden_layer1)
    model = Model(inputs=input, outputs=out_layer)
    return model

def model_sswe_u(window_size, vocab_size, embedding_size):
    input_original = Input(shape=(window_size,), dtype='int32', name='input_original')
    input_corrupt = Input(shape=(window_size,), dtype='int32', name='input_corrupt')

    embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size, input_length=window_size, name='embedding')
    concat_layer = Flatten(name='concatnate')
    hidden_layer1 = Dense(20, activation='tanh', name='hidden')
    out_layer = Dense(2, name='output')

    embedding_main = embedding(input_original)
    concat_layer_main = concat_layer(embedding_main)
    hidden_main = hidden_layer1(concat_layer_main)
    out_main = out_layer(hidden_main)

    embedding_corrupt = embedding(input_corrupt)
    concat_layer_corrupt = concat_layer(embedding_corrupt)
    hidden_corrupt = hidden_layer1(concat_layer_corrupt)
    out_corrupt = out_layer(hidden_corrupt)
    output = subtract([out_main, out_corrupt])

    model = Model(inputs=[input_original, input_corrupt], outputs=output)
    return model

def custom_loss(alpha = 0.5):
    def sswe_u_loss(y_true, y_pred):
        loss = tf.keras.backend.mean(tf.maximum(1. - y_pred*y_true, 0.) * tf.constant([2*alpha, 2-2*alpha]))
        return loss
    return sswe_u_loss


def run_sswe_h(window_size, training_file, vocab_size, embedding_size):
    model = model_sswe_h(window_size, vocab_size, embedding_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file, vocab_size)
    one_hot_labels = to_categorical(labels, num_classes=2)
    model.fit(inputs, one_hot_labels, epochs=100, batch_size=5000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    np.save('word_embedding.npy', weights)
    print(weights.shape)
    return weights

def run_sswe_u(window_size, training_file, vocab_size, embedding_size, alpha=0.5, num_negative_samples=15):
    model = model_sswe_u(window_size, vocab_size, embedding_size)
    sswe_u_loss = custom_loss(alpha=alpha)
    model.compile(optimizer=Adagrad(lr=0.01), loss=sswe_u_loss, metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file, vocab_size, num_negative_samples=num_negative_samples)
    print(labels.shape)

    model.fit([inputs[:,0,:], inputs[:,1,:]], labels, epochs=100, batch_size=10000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    np.save('word_embedding.npy', weights)
    return weights

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
    # Setting parameters
    window_size = 3
    embedding_size = 50
    alpha = 0.5
    num_negative_samples = 15
    training_file = 'data/training_data_real_vocab.txt'
    test_file = 'data/test_data_real_vocab.txt'
    vocab_file = 'data/real_vocab.txt'

    with open(vocab_file) as fp:
        vocab_size = len(fp.readlines())

    #word_embedding = run_sswe_u(window_size, training_file, vocab_size, embedding_size,
    #                            alpha=alpha, num_negative_samples=num_negative_samples)
    #word_embedding = run_sswe_h(window_size, training_file, vocab_size, embedding_size)
    word_embedding = np.load('word_embedding.npy')
    new_data = represent_doc_embedding(training_file, test_file, word_embedding=word_embedding, type='concat')
    #for c in np.arange(0.5,2,0.1):
        #print(c)
    do_classification(new_data, C=1.0)






