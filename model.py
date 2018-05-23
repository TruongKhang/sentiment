import sys
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import to_categorical

def get_model(window_size):
    input = Input(shape=(window_size,), dtype='int32', name='input')
    embedding = Embedding(output_dim=50, input_dim=202115, input_length=window_size, name='embedding')(input)
    concat_layer = Flatten(name='concatnate')(embedding)
    hidden_layer1 = Dense(20, activation='tanh', name='hidden')(concat_layer)
    out_layer = Dense(2, activation='softmax', name='output')(hidden_layer1)
    model = Model(inputs=input, outputs=out_layer)
    return model

def load_dataset(window_size, training_file):
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
            sequences.append(list_words)
            labels.append(label)

        line = fp.readline()
    fp.close()

    return (np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32))

def run(window_size, training_file):
    model = get_model(window_size)
    model.compile(optimizer=Adagrad(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    inputs, labels = load_dataset(window_size, training_file)
    one_hot_labels = to_categorical(labels, num_classes=2)
    model.fit(inputs, one_hot_labels, epochs=100, batch_size=1000)
    weights = model.get_layer('embedding').get_weights()[0]
    print(weights.shape)

if __name__ == '__main__':
    window_size = 3 #sys.argv[1]
    training_file = 'data/training_data_sq.txt'
    run(window_size, training_file)





