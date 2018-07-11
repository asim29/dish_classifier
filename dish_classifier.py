import numpy as np
import pandas as pd 
from time import time 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle

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


# print df
dishes, labels = ParseAndShuffleData(filename)

# print dishes
# print labels
t0 = time()	
vect = CountVectorizer()
features = vect.fit_transform(dishes)
print "vectorize time:", round(time() - t0, 3), "s"

# print vect.get_feature_names()
print features.shape
