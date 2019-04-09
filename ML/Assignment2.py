# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 21:26:20 2019

@author: Atharva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_excel("dataset.xlsx")
X=dataset.iloc[:, 1:5].values
y=dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le_y= LabelEncoder()
y=le_y.fit_transform(y)

le_x= LabelEncoder()
le_x1= LabelEncoder()
le_x2= LabelEncoder()
le_x3= LabelEncoder()
X[:,0]= le_x.fit_transform(X[:, 0])
X[:, 1]=le_x1.fit_transform(X[:, 1])
X[:, 2]=le_x2.fit_transform(X[:, 2])
X[:, 3]=le_x3.fit_transform(X[:, 3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion="entropy")
classifier.fit(X_train, y_train)

y_pred= classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))