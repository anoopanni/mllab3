import math
import numpy as np
from scipy import linalg
import pylab as pl

def lowers(x, y):
    n = 100
    r = 25
    h = [np.sort(np.abs(x-x[i]))[r] for i in range(n)]
    myarray=np.abs((x[:,None]-x[None,:])/h)
    w = np.clip(myarray,0.0,1.0) 
    w = (1 - w*3)*3
    yest = np.zeros(n)
    delta = np.ones(n)
    for i in range(n):
        weights = delta * w[:,i]
        b = np.array([np.sum(weights*y),np.sum(weights*y*x)])
        A = np.array([[np.sum(weights),np.sum(weights*x)],[np.sum(weights*x),np.sum(weights*x*x)]])
        beta = linalg.solve(A,b)
        yest[i] = beta[0] + beta[1]*x[i]
    return yest

def main():
    n = 100
    x = np.linspace(0, 2 * math.pi, n)
    print("====================== value of x =========================")
    print(x)
    y = np.sin(x) + 0.3 * np.random.randn(n)
    print("====================== value of y =========================")
    print(y)
    
    yest = lowers(x,y)


    pl.plot(x,y, label = 'y noisy')
    pl.plot(x,yest, label = 'y pred')
    pl.legend()
    pl.show()
main()