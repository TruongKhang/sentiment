import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import lil_matrix

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

def represent_doc_embedding(training_file, test_file, word_embedding=None, type='concat'):
    print('Representing document...')
    corpus_train = []
    corpus_test = []
    num_training_doc = 0
    num_test_doc = 0
    y_train, y_test = list(), list()
    for i, file in enumerate([training_file, test_file]):
        fp = open(file)
        line = fp.readline()
        while line:
            label_ids = list(map(int, line.strip().split()))
            if i==0:
                num_training_doc += 1
                corpus_train.append(label_ids[1:])
                y_train.append(label_ids[0])
            else:
                num_test_doc += 1
                corpus_test.append(label_ids[1:])
                y_test.append(label_ids[0])
            line = fp.readline()
        fp.close()

    converter = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
    tfidf_train = converter.fit_transform(corpus_train)
    tfidf_test = converter.transform(corpus_test)
    mapping = {v: k for k, v in converter.vocabulary_.items()}
    del corpus_train, corpus_test

    if (type == 'concat') or (type == 'average'):
        if type == 'concat':
            vocab_size, embedding_size = word_embedding.shape
            X_train = lil_matrix((num_training_doc, embedding_size*vocab_size), dtype=np.float64)
            X_test = lil_matrix((num_test_doc, embedding_size*vocab_size), dtype=np.float64)
        else:
            vocab_size, embedding_size = word_embedding.shape
            X_train = np.zeros((num_training_doc, embedding_size))
            X_test = np.zeros((num_test_doc, embedding_size))
        #from time import time

        for d in range(num_training_doc):
            cols = tfidf_train[d].tocoo().col
            indexs = list(map(lambda x: mapping[x], cols))
            tfidf_values = tfidf_train[d].tocoo().data[:, np.newaxis]
            word_embedding_in_d = tfidf_values * word_embedding[indexs]
            if type == 'concat':
                new_cols = np.array(list(map(lambda x: np.arange(x*embedding_size,(x+1)*embedding_size), indexs)), dtype=np.int32)
                new_cols = new_cols.flatten()
                new_rows = np.zeros(len(cols)*embedding_size, dtype=np.int32) + d
                X_train[new_rows, new_cols] = word_embedding_in_d.flatten()

            else:
                vector = np.sum(word_embedding_in_d, axis=0) / len(cols)
                X_train[d] = vector

        for d in range(num_test_doc):
            cols = tfidf_test[d].tocoo().col
            if len(cols) > 0:
                #print(d, len(cols))
                indexs = list(map(lambda x: mapping[x], cols))
                tfidf_values = tfidf_test[d].tocoo().data[:, np.newaxis]
                word_embedding_in_d = tfidf_values * word_embedding[indexs]
                if type == 'concat':
                    new_cols = np.array(
                        list(map(lambda x: np.arange(x * embedding_size, (x + 1) * embedding_size), indexs)),
                        dtype=np.int32)
                    new_cols = new_cols.flatten()
                    new_rows = np.zeros(len(cols) * embedding_size, dtype=np.int32) + d
                    X_test[new_rows, new_cols] = word_embedding_in_d.flatten()

                else:
                    vector = np.sum(word_embedding_in_d, axis=0) / len(cols)
                    X_test[d] = vector
    else:
        X_train, X_test = tfidf_train, tfidf_test

    return X_train, y_train, X_test, y_test
