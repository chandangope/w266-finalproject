from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import collections
import zipfile

import numpy as np



def download_glove(output_dir="data"):
    import wget
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # NOTE: these are uncased vectors from 6B tokens from Wikipedia + Gigaword
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    return wget.download(url, out=output_dir)
    
def archive_line_iter(archive_path, inner_path):
    with zipfile.ZipFile(archive_path) as arx:
        with arx.open(inner_path) as fd:
            for line in fd:
                yield line

def parse_glove_file(archive_path, ndim):
    # File path inside archive
    inner_path = "glove.6B.{:d}d.txt".format(ndim)
    print("Parsing file: {:s}:{:s}".format(archive_path, inner_path))
    # Count lines to pre-allocate memory
    line_count = 0
    for line in archive_line_iter(archive_path, inner_path):
        line_count += 1
    print("Found {:,} words.".format(line_count))
    
    # Pre-allocate vectors as a contiguous array
    # Add three for for <s>, </s>, and <unk>
    W = np.zeros((3+line_count, ndim), dtype=np.float32)
    words = ["<s>", "</s>", "<unk>"]

    print("Parsing vectors... ", end="")
    line_iter = archive_line_iter(archive_path, inner_path)
    for i, line in enumerate(line_iter):
        word, numbers = line.split(maxsplit=1)
        words.append(word.decode('utf-8'))
        W[3+i] = np.fromstring(numbers, dtype=np.float32, sep=" ")
    print("Done! (W.shape = {:s})".format(str(W.shape)))
    return words, W


class Hands(object):
    """Helper class to manage GloVe vectors."""
    
    _AVAILABLE_DIMS = { 50, 100, 200, 300 }

    def __init__(self, ndim=50):
        assert(ndim in self._AVAILABLE_DIMS)

        self.vocab = None
        self.W = None
        self.zipped_filename = "data/glove/glove.6B.zip"

        # Download datasets
        if not os.path.isfile(self.zipped_filename):
            data_dir = os.path.dirname(self.zipped_filename)
            print("Downloading GloVe vectors to {:s}".format(data_dir))
            self.zipped_filename = download_glove(data_dir)
        print("Loading vectors from {:s}".format(self.zipped_filename))

        words, W = parse_glove_file(self.zipped_filename, ndim)
        self.vocab = words
        # Set nonzero value for special tokens
        mean_vec = np.mean(W[3:], axis=0)
        for i in range(3):
            W[i] = mean_vec
        self.W = W
        self.word_to_id = dict(zip(words, np.arange(len(words))))
        self.id_to_word = dict(zip(np.arange(len(words)), words))


    @property
    def shape(self):
        return self.W.shape

    @property
    def nvec(self):
        return self.W.shape[0]

    @property
    def ndim(self):
        return self.W.shape[1]
    
    def get_vector(self, word, strict=False, verbose=False):
        """Get the vector for a given word. If strict=True, will not replace 
        unknowns with <unk>."""
        if strict: 
            assert word in self.vocab, "Word '{:s}' not found in vocabulary.".format(word)
        if word not in self.vocab:
            if verbose:
                print("Word {0} not found, replacing with '<unk>'".format(word))
            word = "<unk>"

        id = self.word_to_id[word]
        assert(id >= 0 and id < self.W.shape[0])
        return self.W[id]

    def find_nn_cos(v, Wv, k=10):
	    """Find nearest neighbors of a given word, by cosine similarity.
	    
	    Returns two parallel lists: indices of nearest neighbors, and 
	    their cosine similarities. Both lists are in descending order, 
	    and inclusive: so nns[0] should be the index of the input word, 
	    nns[1] should be the index of the first nearest neighbor, and so on.
	    
	    Args:
	      v: (d-dimensional vector) word vector of interest
	      Wv: (V x d matrix) word embeddings
	      k: (int) number of neighbors to return
	    
	    Returns (nns, ds), where:
	      nns: (k-dimensional vector of int), row indices of nearest neighbors, 
	        which may include the given word.
	      similarities: (k-dimensional vector of float), cosine similarity of each 
	        neighbor in nns.
	    """
	    Wv_norm = np.linalg.norm(Wv,axis=1)
	    v_norm = np.linalg.norm(v)
	    norm_product = np.multiply(Wv_norm, v_norm) #Wv_norm * v_norm
	    cosine_similarity = np.divide(np.dot(Wv,v), norm_product)   #np.dot(Wv,v) / norm_product
	    sorted_ind = np.argsort(1-cosine_similarity)
	    nns = sorted_ind[:k]
	    similarities = cosine_similarity[nns]
	    return nns, similarities

    def analogy(vA, vB, vC, Wv, k=5):
        """
        Find the vector(s) that best answer "A is to B as C is to ___", returning 
        the top k candidates by cosine similarity.
        
        Args:
          vA: (d-dimensional vector) vector for word A
          vB: (d-dimensional vector) vector for word B
          vC: (d-dimensional vector) vector for word C
          Wv: (V x d matrix) word embeddings
          k: (int) number of neighbors to return

        Returns (nns, ds), where:
          nns: (k-dimensional vector of int), row indices of the top candidate 
            words.
          similarities: (k-dimensional vector of float), cosine similarity of each 
            of the top candidate words.
        """
        v = vB-vA + vC
        nns, similarities = find_nn_cos(v, Wv, k)
        return nns, similarities


    def __getitem__(self, word):
        return self.get_vector(word, verbose=True)



if __name__ == "__main__":
    hands = Hands()
    word = 'chandangope'
    print("Word is {0}".format(word))
    print("Vector = {0}".format(hands[word]))