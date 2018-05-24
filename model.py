import sys
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, subtract
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import to_categorical
import tensorflow as tf

def model_sswe_h(window_size):
    input = Input(shape=(window_size,), dtype='int32', name='input')
    embedding = Embedding(output_dim=50, input_dim=202115, input_length=window_size, name='embedding')(input)
    concat_layer = Flatten(name='concatnate')(embedding)
    hidden_layer1 = Dense(20, activation='tanh', name='hidden')(concat_layer)
    out_layer = Dense(2, activation='softmax', name='output')(hidden_layer1)
    model = Model(inputs=input, outputs=out_layer)
    return model

def model_sswe_u(window_size):
    input_original = Input(shape=(window_size,), dtype='int32', name='input_original')
    input_corrupt = Input(shape=(window_size,), dtype='int32', name='input_corrupt')

    embedding = Embedding(output_dim=50, input_dim=202115, input_length=window_size, name='embedding')
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

def sswe_u_loss(y_true, y_pred, alpha=0.5):

    #print(out_main.type)
    print(tf.shape(y_pred))
    print(tf.shape(y_true))
    syntactic_loss = tf.maximum(0., 1 - y_pred[0])
    sigma = 1
    if y_true[0] == 1:
        sigma = -1
    sentiment_loss = tf.maximum(0., 1 - sigma*y_pred[1])
    return alpha*syntactic_loss * (1-alpha)*sentiment_loss

def load_dataset(window_size, training_file, num_negative_samples=None):
    vocab = None
    corrupt_samples = None
    if num_negative_samples is not None:
        vocab = np.load('data/real_vocab.npy')
        corrupt_samples = np.random.choice(vocab, size=num_negative_samples, replace=False)
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
                    labels.append(label)
            else:
                sequences.append(list_words)
                labels.append(label)


        line = fp.readline()
    fp.close()

    return (np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32))

def run_sswe_h(window_size, training_file):
    model = model_sswe_h(window_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file)
    one_hot_labels = to_categorical(labels, num_classes=2)
    model.fit(inputs, one_hot_labels, epochs=100, batch_size=5000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    print(weights.shape)

def run_sswe_u(window_size, training_file):
    model = model_sswe_u(window_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss=sswe_u_loss, metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file, num_negative_samples=10)
    print(labels.shape)
    model.fit([inputs[:,0,:], inputs[:,1,:]], labels, epochs=100, batch_size=5000, shuffle=True)
    weights = model.get_layer('embedding').get_weights()[0]
    return weights

#def represent_data(word_embedding):


if __name__ == '__main__':
    window_size = 3 #sys.argv[1]
    training_file = 'data/training_data_sq.txt'
    run_sswe_u(window_size, training_file)





