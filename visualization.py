import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
 
 
def main():
 
    embeddings_file = sys.argv[1]
    wv, vocabulary = load_embeddings(embeddings_file)
 
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(wv[:1000,:])
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()
 
 
def load_embeddings(file_name):
 
    wv = np.load(file_name)
    vocab = np.loadtxt('data/real_vocab.txt', dtype='str')
    return wv, vocab
 
if __name__ == '__main__':
    main()
