import numpy as np
import re
import os
import shutil
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_splitted_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    x = positive_examples + negative_examples
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x, y]

def saveMisclassifiedSamples(X, predictions, actual):
    outfolder = 'misclassified'
    if os.path.exists(outfolder):
        shutil.rmtree(outfolder)
    os.makedirs(outfolder)

    falsepos_path = os.path.join(outfolder, "falsepos.txt")
    falseneg_path = os.path.join(outfolder, "falseneg.txt")

    false_pos = 0
    false_neg = 0
    total_pos = 0
    total_neg = 0

    for i,a in enumerate(actual):
        act = a
        pred = predictions[i]
        if act == 0:
            total_neg += 1
        if act == 1:
            total_pos += 1

        if act == 0 and pred > 0.5:
            false_pos += 1
            if os.path.exists(falsepos_path):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            fpos = open(falsepos_path, append_write)
            fpos.write(X[i] + '\n')
            fpos.close();

        if act == 1 and pred < 0.5:
            false_neg += 1
            if os.path.exists(falseneg_path):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
            fneg = open(falseneg_path, append_write)
            fneg.write(X[i] + '\n')
            fneg.close();

    print("Total Pos: {0}, False Negative: {1}, Error%: {2}".format(total_pos, false_neg, 100.0*false_neg/total_pos))
    print("Total Neg: {0}, False Positive: {1}, Error%: {2}".format(total_neg, false_pos, 100.0*false_pos/total_neg))

        
    

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def makeBatches(X, y, batch_size, num_epochs, shuffle=True):
    data_pos = [(X[i], [0,1]) for i, val in enumerate(y) if val == 1]
    data_neg = [(X[i], [1,0]) for i, val in enumerate(y) if val == 0]
    data_pos = np.array(data_pos)
    data_neg = np.array(data_neg)
    data_size_pos = len(data_pos)
    data_size_neg = len(data_neg)
    
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size_pos))
            shuffled_data_pos = data_pos[shuffle_indices]
            shuffle_indices = np.random.permutation(np.arange(data_size_neg))
            shuffled_data_neg = data_neg[shuffle_indices]
        else:
            shuffled_data_pos = data_pos
            shuffled_data_neg = data_neg
        batch_num = 0
        while True:
            start_index = int(batch_num * batch_size/2)
            end_index = int(start_index + batch_size/2)
            if(end_index > data_size_pos or end_index > data_size_neg):
                break
            batch_pos = shuffled_data_pos[start_index:end_index]
            batch_neg = shuffled_data_neg[start_index:end_index]
            batch_num += 1
            yield np.concatenate([batch_pos, batch_neg], 0)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_embedding_vectors_glove(vocabulary, filename='../data/glove/glove.6B/glove.6B.100d.txt', vector_size=100):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
