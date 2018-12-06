# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 07:14:36 2018

@author: DELL
"""

from math import exp
from random import seed
from random import random

def initialize_network(n_inputs,n_hidden,n_outputs):
    network=[]
    hidden_layer=[{'weights':[random() for i in range(n_inputs+1)]}for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer=[{'weights':[random() for i in range(n_hidden+1)]}for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activate(weights,input):
    activation=weights[-1]
    for i in range(len(weights)-1):
        activation+=weights[i]*input[i]
    return activation

def transfer(activation):
    return 1.0/(1.0+exp(-activation))

def forward_propogate(network,row):
    input=row
    for layer in network:
        new_inputs=[]
        for neuron in layer:
            activation=activate(neuron['weights'],input)
            neuron['output']=transfer(activation)
            new_inputs.append(neuron['output'])
        input=new_inputs
    return input

def transfer_derivative(output):
    return output*(1.0-output)

def backward_propogate_error(network,expected):
    for i in reversed(range(len(network))):
        layer=network[i]
        errors= []
        if i!=len(network)-1:
            for j in range(len(layer)):
                error=0
                for neuron in network[i+1]:
                    error+=neuron['weights'][j]*neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron=layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*transfer_derivative(neuron['output'])
                
def update_weights(network,row,l_rate):
    for i in range(len(network)):
        input=row[:-1]
        if i!=0:
            input=[neuron['output']for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(input)):
                neuron['weights'][j]+=l_rate*neuron['delta']*input[j]
                neuron['weights'][-1]+=l_rate*neuron['delta']
                
def training_network(network,train,l_rate,n_epoch,n_output):
    for epoch in range(n_epoch):
        sum_error=0
        for row in train:
            outputs=forward_propogate(network,row)
            expected=[0 for i in range(n_outputs)]
            expected[row[-1]]=1
            sum_error+=sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propogate_error(network,expected)
            update_weights(network,row,l_rate)
            print('>epoch=%d,lrate=%.3f,error=%.3f'%(epoch,l_rate,sum_error))
            
seed(1)            
dataset=[[1.234534,2.234234232,0],
         [2.234534342,2.564234232,0],
         [3.234534345,2.234234232,0],
         [1.234534231,1.834234232,0],
         [2.234534453,3.234234232,0],
         [3.234534112,4.234234232,1],
         [3.234534456,3.934234232,1],
         [1.234534345,1.534234232,1],
         [1.234534332,-0.234234232,1],
         [1.234534122,1.434234232,1]]
n_inputs=len(dataset[0])-1
n_outputs=len(set([row[-1]for row in dataset]))  
network=initialize_network(n_inputs,2,n_outputs)
training_network(network,dataset,0.5,20,n_outputs)
for layer in network:
    print(layer)
    