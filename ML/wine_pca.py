# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:05:57 2019

@author: Atharva
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine


X, y= load_wine(return_X_y=True)

Y=pd.DataFrame(y)

from sklearn.preprocessing import StandardScaler
X= StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pcs= pca.fit_transform(X)

new_df= pd.DataFrame(data= pcs, columns=['PC1','PC2'])

final_df= pd.concat([new_df, Y], axis=1)
plt.scatter(new_df['PC1'],new_df['PC2'])
plt.show()
#pca.explained_variance_ratio_   for variance captured
#pca.explained_variance_ratio_.cumsum()   for variance captured by the each of the no of pc