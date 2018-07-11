import numpy as np
import pandas as pd 
from time import time 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn import model_selection	
from sklearn.svm import SVC

filename = "test.csv"

def ParseAndShuffleData(filename):
	df = pd.read_csv(filename)
	df.columns = ['Dish', 'Category']

	category_list = []
	for index, row in df.iterrows():
		row['Category'] = row['Category'].lower()
		size = len(row['Category'])
	# Deal with spaces at beginning or end
		if row['Category'][0] == ' ':
			row['Category'] = row['Category'][1:size]
			size -= 1
		if row['Category'][size-1] == ' ':
			row['Category'] = row['Category'][0:size-1]

		if row['Category'] not in category_list:
			category_list.append(row['Category'])

	category_list = np.array(category_list)

	df = shuffle(df, random_state = 0)
	dishes = np.array([x.lower() for x in np.array(df['Dish'])])
	labels = np.array(df['Category'])

	return dishes, labels
	# d_cat.to_csv("categories.csv")

def PreProcess(features, labels):
	dishes_train, dishes_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.1, random_state = 1)
	t0 = time()	
	vect = CountVectorizer()
	dishes_train = vect.fit_transform(dishes_train)
	dishes_test = vect.transform(dishes_test)
	print "vectorize time:", round(time() - t0, 3), "s"
	print dishes_train.shape
	print dishes_test.shape

	return dishes_train, dishes_test, labels_train, labels_test
		
def SVM(dishes_train, dishes_test, labels_train, labels_test):
	clf = SVC()
	clf.fit(dishes_train, dishes_test)
	clf.predict(dishes_test)
# print df

dishes, labels = ParseAndShuffleData(filename)
dishes_train, dishes_test, labels_train, labels_test = PreProcess(dishes, labels)
# print dishes
# print labels

# print vect.get_feature_names()
