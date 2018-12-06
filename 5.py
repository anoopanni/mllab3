# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:31:15 2018

@author: DELL
"""

import math
import csv

def safe_div(x,y):
    if y==0:
        return 0
    return x/y

def loadCSV(filename):
    lines=csv.reader(open(filename))
    dataset =[]
    for d in lines:
        dataset.append(d)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset
def splitdataset(dataset,splitratio):
    trainsize=int(len(dataset)*splitratio)
    trainset=[]
    copy=[]
    for c in dataset:
        copy.append(c)
    i=0
    while len(trainset)<trainsize:
        trainset.append(copy.pop(i))
    return[trainset,copy]

def separatedByClass(dataset):
    separated={}
    for i in range(len(dataset)):
        vector=dataset[i]
        if(vector[-1] not in separated):
            separated[vector[-1]]=[]
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
    return safe_div(sum(numbers),float(len(numbers)))

def stdev(numbers):
    avg=mean(numbers)
    variance=safe_div(sum([pow(x-avg,2)for x in numbers]),float(len(numbers)-1))
    return math.sqrt(variance)

def summarize(dataset):
    summaries=[(mean(attribute),stdev(attribute))for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated=separatedByClass(dataset)
    summaries={}
    for classvalue,instances in separated.items():
        summaries[classvalue]=summarize(instances)
    return summaries

def calculateprobability(x,mean,stdev):
    exponent=math.exp(-safe_div(math.pow(x-mean,2),(2*math.pow(stdev,2))))
    final=safe_div(1,(math.sqrt(2*math.pi))*stdev)*exponent
    return final

def calculateclassprobability(summaries,inputvector):
    probabilities={}
    for classvalue,classsummaries in summaries.items():
        probabilities[classvalue]=1
        for i in range(len(classsummaries)):
            mean,stdev=classsummaries[i]
            x=inputvector[i]
            probabilities[classvalue]*=calculateprobability(x,mean,stdev)
    return probabilities

def predict(summaries,inputvector):
    probabilities=calculateclassprobability(summaries,inputvector)
    bestlabel,bestprob=None,-1
    for classvalue,probability in probabilities.items():
        if bestlabel is None or probability>bestprob:
            bestprob=probability
            bestlabel=classvalue
    return classvalue

def get_prediction(summaries,testset):
    prediction=[]
    for i in range(len(testset)):
        results=predict(summaries,testset[i])
        prediction.append(results)
    return prediction

def get_accuracy(testset,prediction):
    correct=0
    for i in range(len(testset)):
        if testset[i][-1]==prediction[i]:
            correct+=1
    accuracy=safe_div(correct,float(len(testset)))*100.0
    return accuracy

def main():
    filename='ConceptLearning.csv'
    dataset=loadCSV(filename)
    splitratio=0.5
    trainset,testset=splitdataset(dataset,splitratio)
    print('Split {0}rows into:'.format(len(dataset)))
    print('No of training data is:'+(repr(len(trainset))))
    print('No of testing data set is:'+(repr(len(testset))))
    print("Values assumed for the concept learning attributes are\n")
    print("OUTLOOK => SUNNY=1 OVERCAST=2 RAINY=3,TEMP => HOT=1 MILD=2 COOL=3,HUMIDITY => HIGH=1,NORMAL=2,WIND => WEAK=1,STRONG=2")
    print("target function: PlayTennis=>yes=10,no=5")
    print("the training set is")
    for x in trainset:
        print(x)
    print("the training set is")
    for x in trainset:
        print(x)
    summaries=summarizeByClass(testset)
    prediction= get_prediction(summaries,testset)
    actual=[]
    for i in range(len(testset)):
        vector=testset[i]
        actual.append(vector[-1])
    print('Actual:{0}%'.format(actual))
    print('Predicted:{0}%'.format(prediction))
    accuracy=get_accuracy(testset,prediction)
    print('Accuracy:{0}%'.format(accuracy))
main()    