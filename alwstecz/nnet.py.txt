# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:33:44 2022

@author: RZ
"""

import numpy as np 

def randX(a, b): 
    P = np.random.rand(a, b) 
    return P 

def randn(a, b): 
    w = randX(a, b) * 2.0 - 1.0 
    return w 

def normRows(a): 
    P = a.copy() 
    rows, columns = a.shape 
    for x in range(0, rows): 
        sumSq = 0 
        for y in range(0, columns): 
            v = P[x, y] 
            sumSq += v ** 2.0 
            len = np.sqrt(sumSq) 
            for y in range(0, columns): 
                P[x, y] = P[x, y] / len 
    return P 

def sumsqr(a): 
    rows, columns = a.shape 
    sumSq = 0 
    for x in range(0, rows): 
        for y in range(0, columns): 
            v = a[x, y] 
            sumSq += v ** 2.0 
    return sumSq 

def rands(a, b): 
    w = randX(a, b) * 2.0 - 1.0 
    b = randX(a, 1) * 2.0 - 1.0 
    return w, b 

def nwtan(s, p): 
    magw = 0.7 * s ** (1.0 / p) 
    w = magw * normRows(randn(s,p)) 
    b = magw * randn(s,1) 
    rng = np.zeros((1, p)) 
    rng = rng + 2.0 
    mid = np.zeros((p, 1)) 
    w = 2.0 * w / np.dot(np.ones((s,1)), rng) 
    b = b - np.dot(w, mid) 
    return w, b 

def nwlog(s, p): 
    magw = 2.8 * s ** (1.0 / p) 
    w = magw * normRows(randn(s,p))
    b = magw * randn(s,1) 
    rng = np.zeros((1, p)) 
    rng = rng + 2.0 
    mid = np.zeros((p, 1)) 
    w = 2.0 * w / np.dot(np.ones((s,1)), rng) 
    b = b - np.dot(w, mid) 
    return w, b 

def tansig(n, b): 
    n = n + b 
    a = 2.0 / (1.0 + np.exp(-2.0 * n)) - 1.0 
    rows, columns = a.shape 
    for x in range(0, rows): 
        for y in range(0, columns): 
            v = a[x, y] 
            if np.abs(v) == np.inf: 
                a[x, y] = np.sign(n[x, y]) 
    return a 
            
def logsig(n, b): 
    n = n + b 
    a = 1.0 / (1.0 + np.exp(-n)) 
    rows, columns = a.shape 
    for x in range(0, rows): 
        for y in range(0, columns): 
            v = a[x, y] 
            if np.abs(v) == np.inf: 
                a[x, y] = np.sign(n[x, y]) 
    return a 
            
def purelin(n, b): 
    a = n + b 
    return a 

def deltatan(a, d, *w):
    if not w:
        d = (1.0 - (a * a)) * d 
    else:
        d = (1.0 - (a * a)) * np.dot(np.transpose(w[0]), d)
    return d 

def deltalog(a, d, *w): 
    if not w:
        d = a * (1.0 - a) * d 
    else:
        d = a * (1.0 - a) * np.dot(np.transpose(w[0]), d)
    return d 

def deltalin(a, d): 
    return d 

def learnbp(x, d, lr): 
    d = lr * d 
    dw = np.dot(d, np.transpose(x)) 
    Q = x.shape[1] 
    db = np.dot(d, np.ones((Q, 1))) 
    return dw, db
