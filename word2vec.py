import sys
import gensim
import numpy as np
def word2vec(training_file, vocab_size, window_size, embedding_size):
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
    model = gensim.models.Word2Vec(corpus_train, window=window_size, size=embedding_size, min_count=1, workers=4)
    model.train(corpus_train, total_examples=model.corpus_count, epochs=100)
    word_embedding = np.zeros((vocab_size, embedding_size))
    for i in range(vocab_size):
        if str(i) in list(model.wv.vocab):
            word_embedding[i] = model.wv[str(i)]
    np.save('word2vec.npy', word_embedding)

if __name__ == '__main__':
    # Setting parameters
    window_size = int(sys.argv[1])
    embedding_size = int(sys.argv[2])
    training_file = 'data/training_data_real_vocab.txt'
    test_file = 'data/test_data_real_vocab.txt'
    vocab_file = 'data/real_vocab.txt'

    with open(vocab_file) as fp:
        vocab_size = len(fp.readlines())
    word2vec(training_file, vocab_size, window_size, embedding_size)
