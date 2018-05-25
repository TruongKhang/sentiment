import sys, os
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, subtract
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from sklearn.svm import LinearSVC

def model_sswe_h(window_size, vocab_size):
    input = Input(shape=(window_size,), dtype='int32', name='input')
    embedding = Embedding(output_dim=50, input_dim=vocab_size, input_length=window_size, name='embedding')(input)
    concat_layer = Flatten(name='concatnate')(embedding)
    hidden_layer1 = Dense(20, activation='tanh', name='hidden')(concat_layer)
    out_layer = Dense(2, activation='softmax', name='output')(hidden_layer1)
    model = Model(inputs=input, outputs=out_layer)
    return model

def model_sswe_u(window_size, vocab_size):
    input_original = Input(shape=(window_size,), dtype='int32', name='input_original')
    input_corrupt = Input(shape=(window_size,), dtype='int32', name='input_corrupt')

    embedding = Embedding(output_dim=50, input_dim=vocab_size, input_length=window_size, name='embedding')
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

def sswe_u_loss(y_true, y_pred):
    alpha = 0.5
    #print(y_pred.get_shape().as_list())
    #print(y_true.get_shape().as_list())
    loss = tf.keras.backend.mean(tf.maximum(1. - y_pred*y_true, 0.) * tf.constant([2*alpha, 2-2*alpha]))
    return loss

def load_dataset(window_size, training_file, vocab_size, num_negative_samples=None):
    vocab_ids = None
    corrupt_samples = None
    if num_negative_samples is not None:
        vocab_ids = np.arange(vocab_size)
        corrupt_samples = np.random.choice(vocab_ids, size=num_negative_samples, replace=False)
    sequences = []
    labels = []
    fp = open(training_file)
    line = fp.readline()
    while line:
        line = line.strip().split()
        label = int(line[0])
        num_words = len(line) - 1
        for i in range(1, num_words+1):
            start_index = i - 1 - int(window_size/2)
            end_index = i - 1 + int(window_size/2)
            list_words = []
            for j in range(start_index, end_index+1):
                if j < 0:
                    list_words.append(int(line[1]))
                elif j > num_words-1:
                    list_words.append(int(line[num_words]))
                else:
                    list_words.append(int(line[j+1]))
            if num_negative_samples is not None:
                for j in range(num_negative_samples):
                    tr = list(list_words)
                    tr[int(window_size/2)] = corrupt_samples[j]
                    sequences.append([list_words, tr])
                    if label == 0:
                        labels.append([1,1])
                    else:
                        labels.append([1,-1])
            else:
                sequences.append(list_words)
                labels.append(label)
        line = fp.readline()
    fp.close()

    return (np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32))

def run_sswe_h(window_size, training_file, vocab_size):
    model = model_sswe_h(window_size, vocab_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file, vocab_size)
    one_hot_labels = to_categorical(labels, num_classes=2)
    model.fit(inputs, one_hot_labels, epochs=100, batch_size=5000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    np.save('word_embedding.npy', weights)
    print(weights.shape)
    return weights

def run_sswe_u(window_size, training_file, vocab_size):
    model = model_sswe_u(window_size, vocab_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss=sswe_u_loss, metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file, vocab_size, num_negative_samples=10)
    print(labels.shape)
    model.fit([inputs[:,0,:], inputs[:,1,:]], labels, epochs=100, batch_size=10000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    np.save('word_embedding.npy', weights)
    return weights

def represent_data(training_file, test_file, word_embedding, training_new_represent, test_new_represent):
    vocab_size, embedding_size = word_embedding.shape
    fp_out_train = open(training_new_represent, 'w')
    fp_out_test = open(test_new_represent, 'w')

    corpus = []
    num_training_doc = 0
    num_test_doc = 0
    labels = []
    for i, file in enumerate([training_file, test_file]):
        fp = open(file)
        line = fp.readline()
        while line:
            if i==0:
                num_training_doc += 1
            else:
                num_test_doc += 1
            label_ids = list(map(int, line.strip().split()))
            corpus.append(label_ids[1:])
            labels.append(label_ids[0]+1)
            line = fp.readline()
        fp.close()

    converter = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tfidf_matrix = converter.fit_transform(corpus)

    mapping = {v: k for k, v in converter.vocabulary_.items()}
    num_doc, real_vocab_size = tfidf_matrix.shape
    print(num_doc, real_vocab_size)
    for d in range(num_doc):
        if d < num_training_doc:
            fp = fp_out_train
        else:
            fp = fp_out_test
        fp.write('%d ' % labels[d])

        cols = tfidf_matrix[d].tocoo().col
        list_ids = []
        for col in cols:
            #print(col)
            index = mapping[col]
            list_ids.append(index)
        id_sorted = np.argsort(np.array(list_ids))
        for i in id_sorted:
            col = cols[i]
            index = list_ids[i]
            tfidf_value = tfidf_matrix[d, col]
            vector = tfidf_value * word_embedding[index]
            for i, val in enumerate(vector):
                fp.write('%d:%.6f ' %(index*embedding_size+i+1, val))
        fp.write('\n')

def do_classification(training_file, test_file, word_embedding):
    vocab_size, embedding_size = word_embedding.shape

    corpus = []
    num_training_doc = 0
    num_test_doc = 0
    labels = []
    for i, file in enumerate([training_file, test_file]):
        fp = open(file)
        line = fp.readline()
        while line:
            if i==0:
                num_training_doc += 1
            else:
                num_test_doc += 1
            label_ids = list(map(int, line.strip().split()))
            corpus.append(label_ids[1:])
            labels.append(label_ids[0])
            line = fp.readline()
        fp.close()

    converter = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tfidf_matrix = converter.fit_transform(corpus)
    mapping = {v: k for k, v in converter.vocabulary_.items()}

    X_train, y_train = np.zeros((num_training_doc, embedding_size)), np.zeros(num_training_doc)
    X_test, y_test = np.zeros((num_test_doc, embedding_size)), np.zeros(num_test_doc)
    for d in range(len(corpus)):
        if d < num_training_doc:
            y_train[d] = labels[d]
        else:
            y_test[d-num_training_doc] = labels[d]
        

        cols = tfidf_matrix[d].tocoo().col
        for col in cols:
            #print(col)
            index = mapping[col]
            tfidf_value = tfidf_matrix[d, col]
            vector = tfidf_value * word_embedding[index]
            if d < num_training_doc:
                X_train[d] += vector
            else:
                X_test[d-num_training_doc] += vector
        if d < num_training_doc:
            X_train[d] /= len(cols)
        else:
            X_test[d-num_training_doc] /= len(cols)

    svm = LinearSVC(C=0.1)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    np.savetxt('predicted.txt', y_pred+1, fmt='%u')




if __name__ == '__main__':
    window_size = 3 #sys.argv[1]
    training_file = 'data/training_data_sq_real_vocab.txt'
    test_file = 'data/test_data_sq_real_vocab.txt'
    vocab_size = len(np.load('data/real_vocab.npy'))
    #word_embedding = run_sswe_u(window_size, training_file, vocab_size)
    #word_embedding = run_sswe_h(window_size, training_file, vocab_size)
    word_embedding = np.load('word_embedding.npy')
    #represent_data(training_file, test_file, word_embedding, 'data/training_embedding.txt', 'data/test_embedding.txt')
    do_classification(training_file, test_file, word_embedding)
    #os.system('liblinear-1.8/train -s 1 -c 0.1 %s %s' %('data/training_embedding.txt', 'model.txt'))
    #os.system('liblinear-1.8/predict data/test_embedding.txt model.txt predicted.txt')
    os.system('python evaluation.py predicted.txt')






