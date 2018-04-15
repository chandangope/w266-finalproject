#!~/anaconda3/bin/python
# -*- coding: utf-8 -*-

!wget('https://s3.amazonaws.com/danicic-w266-final/train.csv')

print('USE: !python bidirectional_lstm__reluHiSig_softmax_toxic_test.py train.csv <batch size> <epochs>')