from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model

def get_model(window_size):
    input = Input(shape=(window_size,), dtype='int32')
    embedding = Embedding(output_dim=50, input_dim=202115, input_length=window_size)(input)
    concat_layer = Flatten()(embedding)
    hidden_layer1 = Dense(20, activation='tanh')(concat_layer)
    out_layer = Dense(2, activation='softmax')(hidden_layer1)
    model = Model(inputs=input, outputs=out_layer)
    return model

def run(window_size):
    model = get_model(window_size)
    model.compile(optimizer='Adagrad', loss='binary_crossentropy', metrics=['accuracy'])


