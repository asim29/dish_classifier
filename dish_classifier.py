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
import random, logging
import requests
import json
from flask import Flask,request,jsonify

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler())
app.logger.setLevel(logging.INFO)
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

	df = shuffle(df, random_state = 0)
	dishes = np.array(df['Dish'])
	labels = np.array(df['Category'])

	category_list = pd.DataFrame.from_records([category_list]).transpose()
	# print category_list
	category_list.to_csv("categories.csv")
	return dishes, labels

def lemmatize(array):
	stemmer = SnowballStemmer('english')
	lemmatized = []
	for i in range(0, len(array)):
		words = array[i].split()
		words = map(stemmer.stem, words)
		words = ' '.join(words)
		lemmatized.append(words)

	# print array
	return lemmatized

def split(features, labels):
	dishes_train, dishes_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.33, random_state = 1)
	return dishes_train, dishes_test, labels_train, labels_test
		
def train(dishes, labels):

	dishes = lemmatize(dishes)
	vectorizer = TfidfVectorizer(stop_words = 'english')
	vec_dishes = vectorizer.fit_transform(dishes)

	clf = SVC()
	print "Fitting the classifier to the training set"
	# t0 = time()
	param_grid = {
	         'C': [5e2, 1e3, 5e3, 1e4, 5e4],
	          'gamma': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
	          }
	clf = SVC(kernel='rbf', class_weight='balanced', C = 5000, gamma = 0.0005)
	clf = clf.fit(vec_dishes, labels)
	# print "done in %0.3fs" % (time() - t0)

	return clf, vectorizer

def classify(dishes, clf, vectorizer):
	dishes = lemmatize(dishes)
	vec_dishes = vectorizer.transform(dishes)
	prediction = clf.predict(vec_dishes)
	return prediction


@app.route('/classify', methods = ['POST'])
def classifyDish():
	content = request.get_json(force = True)
	if content['category'] != 'UR':
		dishName = content['name'] + ' ' + content['category']
	else:
		dishName = content['name']
	dish = np.array([dishName])
	predicted = classify(dish, clf, vectorizer)
	newCategory = predicted[0]
	newCategory = ' '.join(s[0].upper() + s[1:] for s in newCategory.split())
	dish = {'originalName':content['name'], 'newName':content['name'], 'originalCategory':content['category'], 'newCategory':newCategory}
	result = requests.post('http://restacurant-api/editItem/'+content['id'],params=dish)
	print dish
	return "OK"

@app.route('/health', methods = ['GET'])
def health():
	return "OK";

if __name__ == '__main__':
	dishDict = {}
	t0 = time()
	dishes_train, labels_train = ParseAndShuffleData(filename)
	clf, vectorizer = train(dishes_train, labels_train)
	print "Classifier has been fit"
	app.run(debug = True)

