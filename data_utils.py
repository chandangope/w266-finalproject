
import os
import pandas as pd

class DataLoader(object):
	"""Class to load data"""

	def __init__(self, path="./data"):
		assert(os.path.isdir(path))
		self._path = path

	def readTrainTest(self, verbose=False):
		"""
		Read train.csv and test.csv.

		Args:
		verbose(optional): whether to print basic info while loading

		Returns (train, test), where:
		train: pandas dataframe holding train data
		test: pandas dataframe holding test data
		"""
		train = pd.read_csv(self._path + '/train.csv')
		test = pd.read_csv(self._path + '/test.csv')
		if(verbose==True):
			print("Loaded train.csv: Num rows = {0}, Num cols = {1}".format(train.shape[0], train.shape[1]))
			print("Loaded test.csv: Num rows = {0}, Num cols = {1}".format(test.shape[0], test.shape[1]))
			print("Train data column names: {0}".format(train.columns.get_values()))
			print("Test data column names: {0}".format(test.columns.get_values()))

		return (train, test)