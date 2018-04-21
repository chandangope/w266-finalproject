
import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from random import shuffle

class DataLoader(object):
	"""Class to load data"""

	def __init__(self, path="./data"):
		assert(os.path.isdir(path))
		self._path = path

	def splitDataAndSave(self, data):
	    toxic_no = data.loc[data['toxic'] == 0]['comment_text'].tolist()
	    toxic_yes = data.loc[data['toxic'] == 1]['comment_text'].tolist()
	    toxic_no = [self.clean_str(sent) for sent in toxic_no]
	    toxic_yes = [self.clean_str(sent) for sent in toxic_yes]

	    shuffle(toxic_no)
	    shuffle(toxic_yes)

	    dev_index = -1 * int(0.2 * float(len(toxic_no)))
	    toxic_no_train, toxic_no_dev = toxic_no[:dev_index], toxic_no[dev_index:]
	    print("toxic_no_train size:{0}, toxic_no_dev size:{1}".format(len(toxic_no_train), len(toxic_no_dev)))
	    
	    outF = open("./data/toxic_no_train.txt", "w")
	    for item in toxic_no_train:
	    	outF.write("%s\n" % item)
	    outF.close()

	    outF = open("./data/toxic_no_dev.txt", "w")
	    for item in toxic_no_dev:
	    	outF.write("%s\n" % item)
	    outF.close()


	    dev_index = -1 * int(0.2 * float(len(toxic_yes)))
	    toxic_yes_train, toxic_yes_dev = toxic_yes[:dev_index], toxic_yes[dev_index:]
	    print("toxic_yes_train size:{0}, toxic_yes_dev size:{1}".format(len(toxic_yes_train), len(toxic_yes_dev)))

	    outF = open("./data/toxic_yes_train.txt", "w")
	    for item in toxic_yes_train:
	    	outF.write("%s\n" % item)
	    outF.close()

	    outF = open("./data/toxic_yes_dev.txt", "w")
	    for item in toxic_yes_dev:
	    	outF.write("%s\n" % item)
	    outF.close()




	def readTrainTest(self, verbose=False):
		"""
		Read train.csv and test.csv.

		Args:
		verbose(optional): whether to print basic info while loading

		Returns (train, test), where:
		train: pandas dataframe holding train data
		test: pandas dataframe holding test data
		"""
		train = pd.read_csv(self._path + '/train.csv', dtype={'id': object})
		test = pd.read_csv(self._path + '/test.csv', dtype={'id': object})
		if(verbose==True):
			print("Loaded train.csv: Num rows = {0}, Num cols = {1}".format(train.shape[0], train.shape[1]))
			print("Loaded test.csv: Num rows = {0}, Num cols = {1}".format(test.shape[0], test.shape[1]))
			print("Train data column names: {0}".format(train.columns.get_values()))
			print("Test data column names: {0}".format(test.columns.get_values()))

		return (train, test)

	def clean_str(self, string):
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
	    string = re.sub(" \d+", " NUM", string)
	    string = re.sub(r"\s{2,}", " ", string)
	    return string.strip().lower()

	def readTinyData(self, verbose=False):
		"""
		Read tiny_data.csv

		Args:
		verbose(optional): whether to print basic info while loading

		Returns (tiny_data), where:
		tiny_data: pandas dataframe holding data
		"""
		tiny_data = pd.read_csv(self._path + '/tiny_data.csv', dtype={'id': object})
		if(verbose==True):
			print("Loaded tiny_data.csv: Num rows = {0}, Num cols = {1}".format(tiny_data.shape[0], tiny_data.shape[1]))
			print("Column names: {0}".format(tiny_data.columns.get_values()))

		return tiny_data

	def batchGenerator(self, pd_dataframe, batch_size, num_epochs, verbose=False):
		"""
		Generate batches

		Args:
		'pd_dataframe': pd data frame of format  - 'id' 'comment_text' 'toxic' 'severe_toxic' 'obscene' 'threat' 'insult', 'identity_hate'
		'batch_size': number of rows in a batch
		'num_epochs': number of epochs for which to generate
		'verbose'(optional): whether to print some status messages like epoch completed etc.

		Yields: a batch of 'comments','labels'
		"""


		#Convert pd dataframe to numpy array
		data_np=pd_dataframe.values

		comments = data_np[:, 1] #np array of shape [numrows-in-data,]
		labels = data_np[:, 2:].astype(int) #np array of shape [numrows-in-data, numcols-in-labels]

		dataset = tf.data.Dataset.from_tensor_slices((comments, labels))
		dataset = dataset.shuffle(buffer_size=10000)
		dataset = dataset.batch(batch_size)

		iter = dataset.make_initializable_iterator()
		el = iter.get_next()

		with tf.Session() as sess:
			for epoch in range(num_epochs):
				sess.run(iter.initializer)
				while True:
					try:
						b_comments, b_labels = sess.run(el)
						if(b_comments.shape[0] != batch_size):
							if(verbose==True):
								print("\nInsufficient number of items in batch. Skipping...")
						else:
							yield b_comments, b_labels
					except tf.errors.OutOfRangeError:
						if(verbose==True):
							print("*************Epoch {0} finished*************\n".format(epoch+1))
						break



	def testBatching(self, comments, labels, batch_size, num_epochs):
		dataset = tf.data.Dataset.from_tensor_slices((comments, labels))
		dataset = dataset.shuffle(buffer_size=1000)
		dataset = dataset.batch(batch_size)

		iter = dataset.make_initializable_iterator()
		el = iter.get_next()

		with tf.Session() as sess:
			for epoch in range(num_epochs):
				sess.run(iter.initializer)
				while True:
					try:
						b_comments, b_labels = sess.run(el)
						if(b_comments.shape[0] != batch_size):
							print("\nInsufficient number of items in batch. Skipping...")
						else:
							print("\nGot batch...")
							for c,l in zip(b_comments, b_labels):
								print(c,l)
					except tf.errors.OutOfRangeError:
						print("*************Epoch {0} finished*************\n".format(epoch+1))
						break


			

if __name__ == "__main__":
	data_loader = DataLoader(path="./data")
	train, test = data_loader.readTrainTest(verbose=True)
	# cleaned = data_loader.clean_str( train['comment_text'][16] )
	# print(cleaned)

	data_loader.splitDataAndSave(train)
