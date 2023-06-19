from scipy.cluster import hierarchy
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from FTIR_HCA import HCA
polymerName, waveLength, intensity, polymerID = HCA.parseData2('D4_4_publication5.csv', 2, 1763)

Z = hierarchy.linkage(intensity, 'ward')
f = hierarchy.fcluster(Z, t=12, criterion='distance')
print(f.shape)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
hierarchy.dendrogram(Z, leaf_rotation=90, leaf_font_size=1, labels=polymerName)
plt.show()