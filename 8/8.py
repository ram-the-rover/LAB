 import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture as GM
 iris = datasets.load_iris()
 X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['Target']
 model = KMeans(n_clusters=3)
 model.fit(X)
  KMeans(n_clusters=3)
  scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns = X.columns)
 gmm = GM(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)
 plt.figure(figsize=(14,5))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1,3,1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Target], s=40)
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.subplot(1,3,2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('KMeans Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.subplot(1,3,3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_y], s=40)
plt.title('GMM Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
 Text(0, 0.5, 'Petal Width')
 print('Observation: GMM using EM algorithm based clustering matched the true␣,→labels more closely than the Kmeans.')
