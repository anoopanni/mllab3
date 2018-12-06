# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:51:49 2018

@author: DELL
"""
import csv
import bayespy as bp
from colorama import init
import numpy as np

init()

ageEnum={'SuperSeniorCitizen':0,'SeniorCitizen':1,'MiddleAged':2,'Youth':3,'Teen':4}
genderEnum={'male':0,'Female':1}
familyhistoryEnum={'Yes':0,'No':1}
dietEnum={'High':0,'Medium':1,'Low':2}
lifestyleEnum={'Athlete':0,'Active':1,'Moderate':2,'Sedetary':3}
cholestrolEnum={'High':0,'Border':1,'Low':2}
heartdiseaseEnum={'Yes':0,'No':1}

with open('heart_disease.csv') as f:
    lines=csv.reader(f)
    dataset=list(lines)
    data=[]
    for x in dataset:
        data.append([ageEnum[x[0]],genderEnum[x[1]],familyhistoryEnum[x[2]],dietEnum[x[3]],lifestyleEnum[x[4]],cholestrolEnum[x[5]],heartdiseaseEnum[x[6]]])
        data=np.array(data)
        N=len(data)
    
p_age=bp.nodes.Dirichlet(1*np.ones(5))
age=bp.nodes.Categorical(p_age,plates=(N,))
age.observe(data[:,0])
    
p_gender=bp.nodes.Dirichlet(1*np.ones(2))
gender=bp.nodes.Categorical(p_gender,plates=(N,))
gender.observe(data[:,1])

p_familyhistory=bp.nodes.Dirichlet(1*np.ones(2))
familyhistory=bp.nodes.Categorical(p_familyhistory,plates=(N,))
familyhistory.observe(data[:,2])

p_diet=bp.nodes.Dirichlet(1*np.ones(3))
diet=bp.nodes.Categorcial(p_diet,plates=(N,))
diet.observe(data[:,3])

p_lifestyle=bp.nodes.Dirichlet(1*np.ones(4))
lifestyle=bp.nodes.Categorcial(p_lifestyle,plates=(N,))
lifestyle.observe(data[:,4])

p_cholestrol=bp.nodes.Dirichlet(1*np.ones(3))
cholestrol=bp.nodes.Categorcial(p_cholestrol,plates=(N,))
cholestrol.observe(data[:,5])

p_heartdisease=bp.nodes.Dirichlet(np.ones(2),plates=(5,2,2,3,4,3))
heartdisease=bp.nodes.MultiMixture([age,gender,familyhistory,diet,lifestyle,cholestrol],bp.nodes.Categorical,p_heartdisease)
heartdisease.observe(data[:,6])
p_heartdisease.update()

m=0
while m==0:
    print("\n")
    res=bp.nodes.MultiMixture([int(input('Enter age'+str(ageEnum))),int(input('Enter gender'+str(genderEnum))),int(input('Enter familyhistory:'+str(familyhistoryEnum))),int(input('Enter diet'+str(dietEnum))),int(input('Enter lifestyle'+str(lifestyleEnum))),int(input('Enter cholestrol:'+str(cholestrolEnum))),int(input('Enter heartdisease'+str(heartdiseaseEnum)))],bp.nodes.Categorical,p_heartdisease).get_moments()[0][p_heartdisease['true']]
    print('Probability(heartdisease):'+str(res))
    m=int(input("continue:0 exit:1"))            