
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("weather.csv")

outLook = {'sunny':0, 'overCast':1, 'rainy':2}
temperature = {'hot':0, 'cool':1, 'mild':2, }
humidity = {'high':0, 'normal':1}
wind = {'weak':0, 'strong':1, }
#play = {'yes':0, 'no':1}

data['outLook'] = data['outlook'].map(outLook)
data['temperature'] = data['temperature'].map(temperature)
data['humidity'] = data['humidity'].map(humidity)
data['wind'] = data['wind'].map(wind)
#data['play'] = data['play'].map(play)

print(data.info)

X = data[['outLook','temperature','humidity','wind']].values
print(X[0:14])
Y = data['play']
print(Y[0:14])


from sklearn.model_selection import train_test_split
X_trainset, X_testset, Y_trainset, Y_testset = train_test_splittrain_X, test_X, train_y, test_y= train_test_split(X,Y, test_size=0.5, random_state=0)

SpeciesTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
SpeciesTree

# X_trainset=data[cotdt]
SpeciesTree.fit(X_trainset, Y_trainset)
predTree = SpeciesTree.predict(X_testset)
print(predTree [0:14])
print(Y_testset[0:14])

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Khởi tạo mô hình cây quyết định
SpeciesTree = DecisionTreeClassifier()

# Sử dụng dữ liệu và nhãn
X = data[['outLook','temperature','humidity','wind']].values
Y = data['play'].values

# Fit mô hình vào dữ liệu
SpeciesTree.fit(X, Y)
fn = data.columns[1:5]
cn = list(map(str, data["play"].unique()))

# Khởi tạo figure và plot cây quyết định
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (5, 5), dpi = 100)
tree.plot_tree(SpeciesTree, feature_names = fn, class_names = cn, filled = True)
plt.show()
# Lưu ảnh
fig.savefig('D:/DecisionTreeID3-master/cay.jpg')
print('Done')



