# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 08:33:36 2018

@author: DELL
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups

categories=['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
twenty_train=fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
twenty_test=fetch_20newsgroups(subset='test',categories=categories,shuffle=True)
print(len(twenty_train.data))
print(len(twenty_test.data))
print(twenty_train.target_names)
print("\n".join(twenty_train.data[0].split("n")))
print(twenty_train.target[0])
from sklearn.feature_extraction.text import CountVectorizer
countvector=CountVectorizer()
X_train_tf=countvector.fit_transform(twenty_train.data)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
mod=MultinomialNB()
mod.fit(X_train_tfidf,twenty_train.target)
X_test_tf=countvector.transform(twenty_test.data)
X_test_tfidf=tfidf_transformer.transform(X_test_tf)
predicted=mod.predict(X_test_tfidf)
print("Accuracy:\n",accuracy_score(twenty_test.target,predicted))
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
print("Confusion matrix is:\n",confusion_matrix(twenty_test.target,predicted))