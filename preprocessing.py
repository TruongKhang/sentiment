import sys
from nltk.tokenize import RegexpTokenizer
from collections import Counter

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
                            fp_out.write('%d ' % (label+1))
                            check = False
                        fp_out.write('%d ' %(vocab.index(word)+1))
                if not check:
                    fp_out.write('\n')
            line = fp.readline()
        fp.close()
    fp_out.close()

if __name__ == '__main__':
    train_normal = 'data/training_data/normal_comments.txt' #sys.argv[1]
    train_sara = 'data/training_data/sara_comments.txt' #sys.argv[2]
    test_normal = 'data/test_data/nornal_comments.txt' #sys.argv[3]
    test_sara = 'data/test_data/sara_comments.txt' #sys.argv[4]
    vocab_file = 'data/dicts/id_full.txt' #sys.argv[5]
    stop_words_file = 'data/dicts/stop_words.txt' #sys.argv[6]

    out_train = 'data/training_data_sq.txt'
    out_test = 'data/test_data_sq.txt'
    parsing(train_normal, train_sara, out_train, vocab_file, stop_words_file)
    parsing(test_normal, test_sara, out_test, vocab_file, stop_words_file)
