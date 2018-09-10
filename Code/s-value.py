from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

#print(__doc__)

 
X  = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\new_Mmatrix_mi_me_cnv.csv',sep = '\t',header=None)

y2 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\2_mi_me_cnv.csv',sep = '\t',header=None)
y3 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\3_mi_me_cnv.csv',sep = '\t',header=None )
y4 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\4_mi_me_cnv.csv',sep = '\t',header=None)
y5 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\5_mi_me_cnv.csv',sep = '\t',header=None )
y6 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\6_mi_me_cnv.csv',sep = '\t',header=None )
y7 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\7_mi_me_cnv.csv',sep = '\t',header=None )
y8 = pd.read_table('C:\\Users\\Administrator\\Desktop\\new_data\\SNF_NEW\\LUAD\\8_mi_me_cnv.csv',sep = '\t',header=None )
 
X = X.as_matrix()
#X = np.ones_like(X)-X
X = 1 / X
print(X.shape)
y2 = y2.as_matrix()
y3 = y3.as_matrix()
y4 = y4.as_matrix()
y5 = y5.as_matrix()
y6 = y6.as_matrix()
y7 = y7.as_matrix()
y8 = y8.as_matrix()
print(X.shape)
print(y2.shape)

clusterer = X
#cluster_labels = y2 - 1#.........................................
n_clusters = [2,3,4,5,6,7,8]

cluster_labels = [y2-1,y3-1,y4-1,y5-1,y6-1,y7-1,y8-1]
for i in range(7):
    cluster_labels[i] = cluster_labels[i].ravel()

fig,ax = plt.subplots(2,4)
ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9 = ax.ravel()
fig.set_size_inches(20, 10)
#ax2.set_xlim([-1, 1])
#ax2.set_ylim([0, len(X) + (n_clusters + 1) * 10])

ax = [ax2,ax3,ax4,ax5,ax6,ax7,ax8,]

for i in range(7):
    ax[i].set_xlim([-1, 1])
    ax[i].set_ylim([0, len(X) + (n_clusters[i] + 1) * 10])


#silhouette_avg = silhouette_score(X, cluster_labels)
#print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
#sample_silhouette_values = silhouette_samples(X, cluster_labels)
silhouette_avg = [0,0,0,0,0,0,0]
sample_silhouette_values = [0,0,0,0,0,0,0]

for i in range(0,7):
    silhouette_avg[i] = silhouette_score(X,cluster_labels[i],metric="precomputed")
    print("For n_clusters =", n_clusters[i],"The average silhouette_score is :", silhouette_avg[i])
    sample_silhouette_values[i] = silhouette_samples(X,cluster_labels[i],metric="precomputed")

with open('C:\\Users\\Administrator\\Desktop\\data\\SNF_NEW\\LUAD.txt', 'a') as f:
    f.write('mi_me_cnv')
    f.write('\t')
    for i in range(len(silhouette_avg)):
        f.write(str(silhouette_avg[i]))
        f.write('\t')
    f.write('\n')

#cluster_labels = cluster_labels.ravel()
#print(cluster_labels.shape)

for k in range(7):
    y_lower = 10
    for i in range(n_clusters[k]):

        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them

        ith_cluster_silhouette_values = sample_silhouette_values[k][cluster_labels[k] == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters[k])
        
        ax[k].fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        #ax[k].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax[k].set_title("k=%d"%n_clusters[k])
    ax[k].set_xlabel("The silhouette coefficient values")
    ax[k].set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax[k].axvline(x=silhouette_avg[k], color="red", linestyle="--")

    ax[k].set_yticks([])  # Clear the yaxis labels / ticks
    ax[k].set_xticks([-1,-0.8,-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8,1])
    
    
ax9
plt.plot(n_clusters,silhouette_avg,'b-',marker='h',markerfacecolor='g',markersize = '8',linewidth='2')
ax9.set_title("Silhouette")
ax9.set_xlabel("The number of clusters")
ax9.set_ylabel("Value")
 

plt.savefig("C:\\Users\\Administrator\\Desktop\\picture2\\SNF_NEW\\LUAD\\mi_me_cnv.pdf")
#plt.savefig(".\\picture\\LUAD\\mrna_cnv.pdf")
 
plt.show()

