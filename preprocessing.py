import sys
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import numpy as np

tokenizer = RegexpTokenizer(r'\w+')

def read_vocab_as_dict(file_path):
    fp = open(file_path)
    """line = fp.readline()
    vocab = {}
    index = 0
    while line:
        line = line.strip().split()
        vocab[line[0].lower()] = index
        line = fp.readline()
        index += 1"""
    vocab = map(lambda x: x.strip().split()[0].lower(), fp.readlines())
    fp.close()
    return list(vocab)

def read_stop_words(file_path):
    fp = open(file_path)
    list_stop_words = map(lambda x: x.strip().lower(), fp.readlines())
    fp.close()
    return list_stop_words

def parsing(normal_file, sara_file, out_file, vocab_file, stop_words_file):

    fp_out = open(out_file, 'w')

    vocab = read_vocab_as_dict(vocab_file)
    stop_words = read_stop_words(stop_words_file)

    for label, file in enumerate([normal_file, sara_file]):
        #print(label)
        fp = open(file)
        line = fp.readline()
        while line:
            line = line.strip()
            if len(line) >= 2:

                print(label)
                words = tokenizer.tokenize(line.lower())
                #word_counts = Counter(words)
                #unique_words = word_counts.keys()
                #counts = list(word_counts.values())
                check = True
                for i,word in enumerate(words):
                    if (word not in stop_words) and (word in vocab):
                        #print(word)
                        if check:
                            fp_out.write('%d ' % (label))
                            check = False
                        fp_out.write('%d ' %(vocab.index(word)))
                if not check:
                    fp_out.write('\n')
            line = fp.readline()
        fp.close()
    fp_out.close()

def real_vocab(training_file, test_file):
    vocab = []
    for file in [training_file, test_file]:
        fp = open(file)
        line = fp.readline()
        while line:
            line = line.strip().split()
            for i in range(1, len(line)):
                #id = int(line[i].split(':')[0])
                id = int(line[i])
                if id not in vocab:
                    vocab.append(id)
            line = fp.readline()
    print(len(vocab))
    mapping = {k: v for v, k in enumerate(vocab)}
    for file in [training_file, test_file]:
        fp = open(file)
        fp_out = open(file.split('.')[0]+'_real_vocab.txt', 'w')
        line = fp.readline()
        while line:
            line = line.strip().split()
            fp_out.write('%s ' %line[0])
            for i in range(1, len(line)):
                id = int(line[i])
                fp_out.write('%d ' %(mapping[id]))
            fp_out.write('\n')
            line = fp.readline()
        fp.close()
        fp_out.close()

    np.save('data/real_vocab.npy', np.array(vocab))

if __name__ == '__main__':
    train_normal = 'data/training_data/normal_comments.txt' #sys.argv[1]
    train_sara = 'data/training_data/sara_comments.txt' #sys.argv[2]
    test_normal = 'data/test_data/nornal_comments.txt' #sys.argv[3]
    test_sara = 'data/test_data/sara_comments.txt' #sys.argv[4]
    vocab_file = 'data/dicts/id_full.txt' #sys.argv[5]
    stop_words_file = 'data/dicts/stop_words.txt' #sys.argv[6]

    out_train = 'data/training_data_sq.txt'
    out_test = 'data/test_data_sq.txt'
    #parsing(train_normal, train_sara, out_train, vocab_file, stop_words_file)
    #parsing(test_normal, test_sara, out_test, vocab_file, stop_words_file)
    real_vocab('data/training_data_sq.txt', 'data/test_data_sq.txt')