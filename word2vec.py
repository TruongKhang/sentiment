import gensim
import numpy as np
def word2vec(training_file, vocab_size):
    corpus_train = []
    num_training_doc = 0
    y_train = list()
    fp = open(training_file)
    line = fp.readline()
    while line:
        label_ids = list(map(int, line.strip().split()))
        num_training_doc += 1
        corpus_train.append(label_ids[1:])
        y_train.append(label_ids[0])
        line = fp.readline()
    fp.close()
    corpus_train = list(map(str, corpus_train))
    model = gensim.models.Word2Vec(corpus_train, window=5, size=50, min_count=1, workers=4)
    model.train(corpus_train, total_examples=model.corpus_count, epochs=100)
    word_embedding = np.zeros((vocab_size, 50))
    for i in range(vocab_size):
        if str(i) in list(model.wv.vocab):
            word_embedding[i] = model.wv[str(i)]
        print(i) #word_embedding[i])
    np.save('word2vec.npy', word_embedding)

if __name__ == '__main__':
    vocab_size = len(np.load('data/real_vocab.npy'))
    word2vec('data/training_data_sq_real_vocab.txt', vocab_size)
