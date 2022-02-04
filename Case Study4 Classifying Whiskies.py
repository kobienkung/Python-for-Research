# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:07:52 2021

@author: kobienkung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering

#____________________________________________________________________________
whisky = pd.read_csv('whiskies.txt')
whisky['Region'] = pd.read_csv('regions.txt')
flavors = whisky.iloc[:, 2:14]
corr_flavors = flavors.corr()

plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
#plt.savefig('corr_flavors.pdf')


corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
#plt.axis('tight')
plt.colorbar()
#plt.savefig('corr_whisky.pdf')

#____________________________________________________________________________
model = SpectralCoclustering(n_clusters=6, random_state=0) #6 regions
model.fit(corr_whisky)
model.rows_ #number of row clusters times number of rows
#each row ranges from 0-5
#each column ranges from 0-85 (86 whiskies)
np.sum(model.rows_, axis=1) #True/False for whisky belongs to cluster 0,1,2,3,4,5
model.row_labels_ #each observation belongs to cluster 0,1,2,3,4,5
np.sum(model.rows_, axis=0) #each whisky belongs to 1 cluster

#____________________________________________________________________________
whisky['Group'] = pd.Series(model.row_labels_, index=whisky.index)
whisky = whisky.iloc[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)
correlations = whisky.iloc[:, 2:14].transpose().corr()
correlations = np.array(correlations)

plt.figure(figsize=(14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title('Original')
plt.subplot(122)
plt.pcolor(correlations)
plt.title('Rearranged')
#plt.savefig('correlation.pdf')




