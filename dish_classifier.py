import numpy as np
import pandas as pd 
from time import time 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.utils import shuffle
from sklearn import model_selection	
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

filename = "test.csv"

def ParseAndShuffleData(filename):
	df = pd.read_csv(filename)
	df.columns = ['Dish', 'Category']

	category_list = {}
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
			category_list[row['Category']] = 0
		else:
			category_list[row['Category']] += 1

	# category_list = np.array(category_list)

	df = shuffle(df, random_state = 0)
	dishes = np.array(df['Dish'])
	labels = np.array(df['Category'])

	category_list = pd.DataFrame.from_records([category_list]).transpose()
	# print category_list
	category_list.to_csv("categories.csv")
	return dishes, labels

def lemmatize(array):
	stemmer = SnowballStemmer('english')
	for i in range(0, len(array)):
		words = array[i].split()
		words = map(stemmer.stem, words)
		words = ' '.join(words)
		array[i] = words

	print array
	return array

def split(features, labels):
	dishes_train, dishes_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.33, random_state = 1)
	return dishes_train, dishes_test, labels_train, labels_test

	

def PreProcess(dishes_train, dishes_test):
	t0 = time()	
	vect = TfidfVectorizer(stop_words = 'english')
	vec_dishes_train = vect.fit_transform(dishes_train)
	vec_dishes_test = vect.transform(dishes_test)
	print "vectorize time:", round(time() - t0, 3), "s"
	# print dishes_train.shape
	# print dishes_test.shape
	# print vect.get_feature_names()
	return vec_dishes_train, vec_dishes_test

		
def train(dishes_train, dishes_test, labels_train, labels_test):
	clf = SVC()

	
	print "Fitting the classifier to the training set"
	t0 = time()
	param_grid = {
	         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
	          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
	          }
	clf = SVC(kernel='rbf', class_weight='balanced', C = 5000, gamma = 0.0005)
	clf = clf.fit(dishes_train, labels_train)
	print "done in %0.3fs" % (time() - t0)
	# print "Best estimator found by grid search:"
	# print clf.best_estimator_

	# print pd.DataFrame(labels_test, pred)

	return clf

		
# print df

dishes, labels = ParseAndShuffleData(filename)
dishes = lemmatize(dishes)
dishes_train, dishes_test, labels_train, labels_test = split(dishes, labels)
vec_dishes_train, vec_dishes_test = PreProcess(dishes_train, dishes_test)
clf = train(vec_dishes_train, vec_dishes_test, labels_train, labels_test)

predicted = clf.predict(vec_dishes_test)

final = pd.DataFrame(
	{'Dish': dishes_test,
	'Predicted': predicted,
	'Actual': labels_test})
 
print accuracy_score(labels_test, predicted)
wrong = final[final.Predicted != final.Actual]

wrong.to_csv('Result.csv')
