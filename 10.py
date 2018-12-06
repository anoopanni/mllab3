# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:00:46 2018

@author: DELL
"""

from math import ceil
import numpy as np
from scipy import linalg

def lowers(x,y,f=2./3.,iter=3):
    n=len(x)
    r=int(ceil(n*f))
    h=[np.sort(np.abs(x-x[i]))[r]for i in range(n)]
    w=np.clip(np.abs((x[:,None]-x[None,:])/h),0.0,1.0)
    w=(1-w**3)**3
    yest=np.zeros(n)
    delta=np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights=delta*w[:,i]
            b=np.array([np.sum(weights*y),np.sum(weights*y*x)])
            A=np.array([[np.sum(weights),np.sum(weights*x)],[np.sum(weights*x),np.sum(weights*x*x)]])
            beta=linalg.solve(A,b)
            yest[i]=beta[0]+beta[1]*x[i]
        residuals=y-yest
        s=np.median(np.abs(residuals))
        delta=np.clip(residuals/(s*6.0),-1,1)
        delta=(1-delta**2)**2
    return yest

if __name__=='__main__':
    import math
    n=100
    x=np.linspace(0,2*math.pi,n)
    print("=======Value of X=======")
    print(x)
    y=np.sin(x)+0.3*np.random.randn(n)
    print("=======Value of y=======")
    print(y)
    import pylab as pl
    f=0.25
    yest=lowers(x,y,f=f,iter=3)
    pl.clf()
    pl.plot(x,y,label='x noisy')
    pl.plot(x,yest,label='y pred')
    pl.legend()
    pl.show()