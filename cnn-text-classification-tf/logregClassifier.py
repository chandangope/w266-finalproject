
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

import data_helpers


class LogregClassifier(object):
	def __init__(self):
		self.word_vectorizer = TfidfVectorizer(
		    sublinear_tf=True,
		    strip_accents='unicode',
		    analyzer='word',
		    token_pattern=r'\w{1,}',
		    stop_words='english',
		    ngram_range=(1, 1),
		    max_features=10000)

	def TrainTest(self):
		# Data Preparation
		# ==================================================
		# Load data
		positive_train_file = '../data/toxic_yes_train.txt'
		negative_train_file = '../data/toxic_no_train.txt'
		positive_test_file = '../data/toxic_yes_dev.txt'
		negative_test_file = '../data/toxic_no_dev.txt'

		positive_examples = list(open(positive_train_file, "r").readlines())
		positive_examples = [s.strip() for s in positive_examples]
		negative_examples = list(open(negative_train_file, "r").readlines())
		negative_examples = [s.strip() for s in negative_examples]
		x_train = positive_examples + negative_examples
		positive_labels = [1 for _ in positive_examples]
		negative_labels = [0 for _ in negative_examples]
		y_train = np.concatenate([positive_labels, negative_labels], 0)
		print("x_train length:{0}, y_train shape:{1}".format(len(x_train), y_train.shape))
		print(x_train[0], y_train[0])

		positive_examples = list(open(positive_test_file, "r").readlines())
		positive_examples = [s.strip() for s in positive_examples]
		negative_examples = list(open(negative_test_file, "r").readlines())
		negative_examples = [s.strip() for s in negative_examples]
		x_dev = positive_examples + negative_examples
		positive_labels = [1 for _ in positive_examples]
		negative_labels = [0 for _ in negative_examples]
		y_dev = np.concatenate([positive_labels, negative_labels], 0)
		print("x_dev length:{0}, y_dev shape:{1}".format(len(x_dev), y_dev.shape))
		print(x_dev[-1], y_dev[-1])

		x = x_train+x_dev
		print("x length:{0}".format(len(x)))

		x_test_manual = ["You are the world's biggest moron",
		"I sincerely wish you were not born",
		"Your comments are trash you need to get out of here",
		"Let us not make fun of anyone",
		"Be kind to everyone and not use toxic words like stupid",
		"Be kind to everyone and not use toxic words"]
		y_test_manual = [1, 1, 1, 0, 0, 0, ]

		self.word_vectorizer.fit(x)
		train_word_features = self.word_vectorizer.transform(x_train)
		test_word_features = self.word_vectorizer.transform(x_dev)
		manual_test_word_features = self.word_vectorizer.transform(x_test_manual)

		classifier = LogisticRegression(C=1.0, class_weight='balanced', penalty='l1')
		classifier.fit(train_word_features, y_train)

		predicted = classifier.predict(manual_test_word_features)
		data_helpers.saveMisclassifiedSamples(x_test_manual, predicted, y_test_manual)

		correct_predictions = float(sum(predicted == y_test_manual))
		print("Total number of test examples: {}".format(len(y_test_manual)))
		print("Accuracy: {:g}".format(correct_predictions/float(len(y_test_manual))))

	def Test(self, testData):
		pass



if __name__ == "__main__":
	logregClassifier = LogregClassifier()
	logregClassifier.TrainTest()