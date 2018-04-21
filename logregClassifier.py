
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

import data_helpers


class LogregClassifier(object):
	def __init__(self):
		pass

	def Train(self, trainData, validateData):
		# Data Preparation
		# ==================================================
		# Load data
		print("\nLoading train data...")
		x_train, y_train = data_helpers.load_splitted_data_and_labels('../data/toxic_yes_train.txt', '../data/toxic_no_train.txt')
		print("x_train length:{0}, y_train shape:{1}".format(len(x_train), y_train.shape))
		print(x_train[0], y_train[0])

		print("\nLoading dev data...")
		x_dev, y_dev = data_helpers.load_splitted_data_and_labels('../data/toxic_yes_dev.txt', '../data/toxic_no_dev.txt')
		print("x_dev length:{0}, y_dev shape:{1}".format(len(x_dev), y_dev.shape))
		print(x_dev[-1], y_dev[-1])

		x = x_train+x_dev
		print("x length:{0}".format(len(x)))

	def Test(self, testData):
		pass



if __name__ == "__main__":
	logregClassifier = LogregClassifier()