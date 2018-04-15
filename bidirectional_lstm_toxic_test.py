#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn import model_selection
import numpy as np
import tensorflow  as tf
import keras
import h5py
from keras.models import load_model





#imports:

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


# import data
train_all=pd.read_csv('train.csv')

train_80, test_20 = model_selection.train_test_split(train_all,test_size=0.2)

#preprocessing

max_features = 5000
# cut texts after this number of words, since character max is 5000, word max also has this ceiling

print('Loading data...')

#training data
x_train = train_80['comment_text']
y_train = train_80[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

#testing validation data (not for training model, just validation for )

x_test = test_20['comment_text']
y_test = test_20[['toxic','severe_toxic','obscene','threat','insult','identity_hate']]

# check lengths

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#Tokenize comment_text
Tokenizer= keras.preprocessing.text.Tokenizer(num_words=10000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=False, oov_token=None)

Tokenizer.fit_on_texts(train_all)
x_train=Tokenizer.texts_to_sequences(x_train)
x_test=Tokenizer.texts_to_sequences(x_test)

# parameters
num_classes = 6
batch_size = 50
maxlen = 2300   # longest comment in words split(' ')



print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64,activation='relu',recurrent_activation='hard_sigmoid')))  #relu for activation, hard sigmoid bc fast and reliable
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])

model.save('bidirectional_lstm.h5')