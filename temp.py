# -*- coding: utf-8 -*-

import numpy as np
import math
import random
from scipy.spatial.distance import euclidean
import scipy.stats as ss
import matplotlib.pyplot as plt

#_______________________________________________________________________________________
def majorityVote(votes):
    '''xxx'''
    voteCounts = {}
    for vote in votes:
        if vote in voteCounts:
            voteCounts[vote] += 1
        else:
            voteCounts[vote] = 1

    winner = []
    maxCount = max(voteCounts.values())
    
    for vote, count in voteCounts.items():
        if count == maxCount:
            winner.append(vote)
    'print(winner)'
    return random.choice(winner)


def majorityVoteShort(votes):
    mode, count = ss.mode(votes)
    return mode[0]

votes = [1,3,2,1,2,3,3,3,3,2,2,2]
majorityVoteShort(votes)

#_______________________________________________________________________________________
def findNearestNeighbors(p, points, k=5):
    '''find the k nearest neighbors of point p and return their indices.'''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = euclidean(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]


def knnPredict(p, points, outcome, k=5):
    ind = findNearestNeighbors(p, points, k)
    return majorityVote(outcome[ind])

outcome = np.array([0,0,0,0,1,1,1,1,1])
points = np.array([[1,1],[1,2],[1,3],[2,1],[2,2],[2,3],[3,1],[3,2],[3,3]])
p = np.array([2.5,2])
plt.plot(points[:,0], points[:,1], 'ro')
plt.plot(p[0],  p[1], 'bo')
plt.axis([0.5, 3.5, 0.5, 3.5])
plt.show()

#_______________________________________________________________________________________
def generateSynthData(n=50):
    '''Create 2 sets of points from bivariate normal distributions'''
    points = np.concatenate((ss.norm(0,1).rvs((n,2)), ss.norm(1,1).rvs((n,2))))
    outcomes = np.concatenate((np.repeat(0, n),np.repeat(1, n)))
    return (points, outcomes)

n=20
points, outcomes = generateSynthData(20)                
plt.plot(points[:n,0], points[:n,1], 'ro')
plt.plot(points[n:,0], points[n:,1], 'bo')
plt.show()

#_______________________________________________________________________________________
def makePredictionGrid(predictors, outcomes, limits, h, k):
    '''Classify each point on the prediction grid.'''
    (xMin, xMax, yMin, yMax) = limits
    xs = np.arange(xMin, xMax, h)
    ys = np.arange(yMin, yMax, h)
    xx, yy = np.meshgrid(xs, ys)
    
    predictionGrid = np.zeros(xx.shape, dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x,y])
            predictionGrid[j,i] = knnPredict(p, predictors, outcomes, k)
    
    return (xx, yy, predictionGrid)


def plot_prediction_grid (xx, yy, prediction_grid, filename):
    """ Plot KNN predictions for every point on the grid."""
    from matplotlib.colors import ListedColormap
    background_colormap = ListedColormap (["hotpink","lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red","blue","green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.savefig(filename)
            
predictors, outcomes = generateSynthData()
limits = (-3,4,-3,4)
h = 0.1
k = 50
xx, yy, predictionGrid = makePredictionGrid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, predictionGrid, "knn_Synth_5")

#_______________________________________________________________________________________
from sklearn import datasets
iris = datasets.load_iris()

predictors = iris.data[:,0:2]
outcomes = iris.target

plt.plot(predictors[outcomes==0][:,0], predictors[outcomes==0][:,1], 'ro')
plt.plot(predictors[outcomes==1][:,0], predictors[outcomes==1][:,1], 'go')
plt.plot(predictors[outcomes==2][:,0], predictors[outcomes==2][:,1], 'bo')
plt.show()


limits = (4,8,1.5,4.5)
h = 0.1
k = 5
xx, yy, predictionGrid = makePredictionGrid(predictors, outcomes, limits, h, k)
plot_prediction_grid(xx, yy, predictionGrid, "knn_Synth_5")
plt.show()


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(predictors, outcomes)
skPredictions = knn.predict(predictors)

myPredictions = np.array([knnPredict(p, predictors, outcomes, 5) for p in predictors])

print(np.mean(skPredictions == myPredictions))
print(np.mean(skPredictions == outcomes))
print(np.mean(myPredictions == outcomes))
