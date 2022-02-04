# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:36:06 2021

@author: kobienkung
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#____________________________________________________________________________
n = 50
h = 1
sd = 1

def gen_data(n, h, sd1, sd2):
    x1 = ss.norm.rvs(-h, sd1, n)
    y1 = ss.norm.rvs(0, sd1, n)
    x2 = ss.norm.rvs(h, sd2, n)
    y2 = ss.norm.rvs(0, sd2, n)
    return (x1, y1, x2, y2)

(x1, y1, x2, y2) = gen_data(50, 1, 1, 1.5)

def plot_data(x1, y1, x2, y2):
    plt.figure()
    plt.plot(x1, y1, 'o', ms=2)
    plt.plot(x2, y2, 'o', ms=2)
    plt.xlabel('$x_1$') # '$' makes subscription
    plt.ylabel('$x_2$')


n = 1000
(x1, y1, x2, y2) = gen_data(n, 1.5, 1, 1.5)
plot_data(x1, y1, x2, y2)

#____________________________________________________________________________
clf = LogisticRegression() #classifier
np.vstack((x1,y1)).T.shape # check
X = np.vstack((np.vstack((x1,y1)).T, np.vstack((x2,y2)).T))
X.shape #check
y = np.hstack((np.repeat(1, n), np.repeat(2, n)))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
X_train.shape
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.predict_proba(np.array([-2,0]).reshape(1,-1))
clf.predict_proba([[-2,0],[2,0]]) #alternative
clf.predict(np.array([-2,0]).reshape(1,-1)) #the point belong to class 1 due to greater prob

def plot_prob(ax, clf, class_no):
    xx1, xx2 = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
    probs = clf.predict_proba(np.stack((xx1.ravel(), xx2.ravel()), axis=1))
    z = probs[:, class_no]
    z = z.reshape(xx1.shape)
    CS = ax.contourf(xx1, xx2, z)
    cbar = plt.colorbar(CS)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
plt.figure(figsize=(5,8))
ax = plt.subplot(211)
plot_prob(ax, clf, 0)
plt.title('Predict prob for class 1')
ax = plt.subplot(212)
plot_prob(ax, clf, 1)
plt.title('Predict prob for class 2')
    
    
    
    
    
    




