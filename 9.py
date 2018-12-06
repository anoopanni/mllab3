# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 10:03:32 2018

@author: DELL
"""

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris_dataset=load_iris()
print("\n Iris features \ Target names",iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
    print("\n[{0}]:[{1}]".format(i,iris_dataset.target_names[i]))

X_train,X_test,y_train,y_test=train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0) 

kn=KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,y_train)

x_new=np.array([[5,2.9,1,0.2]])   
print("\n X_new",x_new)

prediction=kn.predict(x_new)

print("Predicted Target value:{}\n".format(prediction))
print("Predicted Feature name:{}\n".format(iris_dataset["target_names"][prediction])) 

i=1
x=X_test[i]
x_new=np.array([x])

for i in range(len(X_test)):
    x=X_test[i]
    x_new=np.array([x])
    prediction=kn.predict(x_new)
    print("\nActual :{0}{1} , Predicted:{2}{3} ".format(y_test[i],iris_dataset["target_names"][y_test[i]],prediction,iris_dataset["target_names"][prediction]))
print ("Test score(accuracy):{:.2f}".format(kn.score(X_test,y_test)))
    