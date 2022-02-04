# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 13:25:06 2021

@author: kobienkung
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#____________________________________________________________________________
n = 100
beta_0 = 5
beta_1 = 2
np.random.seed(1)
x = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x + ss.norm.rvs(loc=0, scale=1, size=n)

plt.figure()
plt.plot(x, y, 'o', ms=5)

xx = np.array([0,10])
plt.plot(xx, beta_0 + beta_1 * xx)
plt.xlabel('x')
plt.ylabel('y')

#____________________________________________________________________________
rss = []
slopes = np.arange(-10, 15, 0.01)
for slope in slopes:
    rss.append(np.sum((y - beta_0 - slope * x) ** 2))

ind_min = np.argmin(rss)
print("Estimate for slope: ", slopes[ind_min])

plt.figure()
plt.plot(slopes, rss,)
plt.xlabel('slopes')
plt.ylabel('RSS')

#____________________________________________________________________________
'''Least Squares Estimaion'''
mod = sm.OLS(y, x)
est = mod.fit()
print(est.summary()) # no intercept(force to start from origin point)

X = sm.add_constant(x) #add interception
mod = sm.OLS(y, X)
est = mod.fit()
print(est.summary())

#____________________________________________________________________________
'''Multiple Linear Regression'''
n = 500
beta_0 = 5
beta_1 = 2
beta_2 = -1
np.random.seed(1)
x_1 = 10 * ss.uniform.rvs(size=n)
x_2 = 10 * ss.uniform.rvs(size=n)
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ss.norm.rvs(loc=0, scale=1, size=n)
X = np.stack([x_1, x_2], axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y, c=y)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')


lm = LinearRegression(fit_intercept=True) # True by default/make sure
lm.fit(X, y)
lm.intercept_ # beta_0
lm.coef_[0] #beta_1
lm.coef_[1] #beta_2

X_0 = np.array([2,4])
lm.predict([[2,4]])
lm.predict([X_0])
lm.predict(X_0.reshape(1,-1))
lm.score(X, y) # R-squared value

#____________________________________________________________________________
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1) #random_state(optional) = randomseed/get identical result
lm = LinearRegression(fit_intercept=True) #unecessary/ True by default
lm.fit(X_train, y_train)
lm.score(X_test, y_test)


















