# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:43:13 2019

@author: Atharva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("kdata.csv")
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y=Y.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder
label_y= LabelEncoder()
Y= label_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.1, random_state=1)

from sklearn.neighbors import KNeighborsClassifier

classifier= KNeighborsClassifier(n_neighbors=3, weights='distance')
classifier.fit(X_train, Y_train)
y_pred= classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, y_pred))

plt.scatter(x=Y_test, y=y_pred)
