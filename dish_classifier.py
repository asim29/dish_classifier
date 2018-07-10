import numpy as np
import pandas as pd 
import time 

filename = "test.csv"

df = pd.read_csv(filename)
df.columns = ['Dish', 'Category']

category_list = []

for index, row in df.iterrows():
	if row['Category'] not in category_list:
		category_list.append(row['Category'])

print np.array(category_list)