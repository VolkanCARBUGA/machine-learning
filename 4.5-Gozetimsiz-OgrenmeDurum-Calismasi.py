import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram,linkage


data=pd.read_pickle("4.5-GozetimsizOgrenmeDurumCalismasi")
X=data.values
X[:,0]=np.abs(2*min(X[:,0]))+X[:,0]
X[:,1]=np.abs(2*min(X[:,1]))+X[:,1]

plt.figure()
plt.scatter(X[:,0],X[:,1],s=50,alpha=0.7, edgecolors="red")
plt.xlabel("Gelir")
plt.ylabel("Harcama")
plt.title("Müşteri Segmentasyonu")
kmeans=KMeans(n_clusters=5)
kmeans.fit(X)
clusters_labels=kmeans.labels_
clusters_centers=kmeans.cluster_centers_
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1],c=clusters_labels,s=50,alpha=0.7, edgecolors="k")
plt.xlabel("Gelir")
plt.ylabel("Harcama")
plt.title("K-Means Kümeleme Sonucu")
linkage_matrix=linkage(X,method="ward")
plt.subplot(1,2,2)
dendrogram(linkage_matrix,labels=clusters_labels)
plt.title("Dendrogram")
plt.xlabel("Müşteri No")
plt.ylabel("Mesafeler")
plt.show()




