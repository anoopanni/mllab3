# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:12:30 2018

@author: DELL
"""

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import pandas as pd

iris=datasets.load_iris()

X=pd.DataFrame(iris.data)
X.columns=['sepal_length','sepal_width','petal_length','petal_width']

Y=pd.DataFrame(iris.target)
Y.columns=['targets']

plt.figure(figsize=(14,7))

colormap=np.array(['Red','Lime','Black'])

plt.subplot(1,2,1)
plt.scatter(X.sepal_length,X.sepal_width,c=colormap[Y.targets],s=40)
plt.title('Sepal')

plt.subplot(1,2,2)
plt.scatter(X.petal_length,X.petal_width,c=colormap[Y.targets],s=40)
plt.title('petal')

plt.figure(figsize=(14,7))

colormap=np.array(['Red','Lime','Black'])

plt.subplot(1,2,1)
plt.scatter(X.petal_length,X.petal_width,c=colormap[Y.targets],s=40)
plt.title('Real Classification')

plt.subplot(1,2,2)
plt.scatter(X.petal_length,X.petal_width,c=colormap[Y.targets],s=40)
plt.title('KMeans Classification')