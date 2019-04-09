# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:20:17 2019

@author: Atharva
"""

#Dataset= Breast Cancer Wisconsin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data= pd.read_csv("data.csv")
X= data.iloc[:, 2:].values
Y= data.iloc[:, 1].values
Y=Y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.20, random_state=4)

from sklearn import svm
sup= svm.SVC()
sup.fit(X_train, Y_train)
print(accuracy_score(Y_test,sup.predict(X_test)))

y_pred= sup.predict(X_test)
