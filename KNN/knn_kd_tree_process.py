# _*_ codig utf8 _*_
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KDTree
import math
a = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
y=np.array([0,1,0,1,1,0])
# a = a[[1,3]]
print(a.var(axis=0))
print(np.median(a,axis=0))

kn =KNeighborsClassifier(n_neighbors=1,leaf_size=2)
kn.fit(a,y.reshape(-1,1))
dist,neighbor = kn.kneighbors([[2,4.5]])
print('neighbor is {}'.format(a[neighbor]))

kdtree = KDTree(a)
neighbors = kdtree.query([[7.5,2]],k=2,return_distance=False)
print(neighbors)

