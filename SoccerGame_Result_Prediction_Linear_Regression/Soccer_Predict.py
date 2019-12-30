"""
=========================
Multi-dimensional scaling
=========================

An illustration of the metric and non-metric MDS on generated noisy data.

The reconstructed points using the metric MDS and non metric MDS are slightly
shifted to avoid overlapping.
"""

# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD

print(__doc__)
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
#################
import pandas as pd
     


from itertools import groupby


def fasta_iter(fasta_name):

    
    fh = open(fasta_name)

# ditch the boolean (x[0]) and just keep the header or sequence since
# we know they alternate.
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
    # drop the ">"
        headerStr = header.__next__()[1:].strip()

    # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)

fiter = fasta_iter('C:\\Users\\LENOVO\\Desktop\\HW4.fas')
print(fiter)
dn=[]
ds=[]
for ff in fiter:
    headerStr, seq = ff
    print(headerStr)
    dn.append(headerStr)
    ds.append(seq)
#    print(seq)

    
print(dn)
print(ds)
d={}
for k in dn:
    for v in ds:
        d[k]=v
print(d)
print(type(headerStr))

df = pd.DataFrame(list(zip(dn, ds)), columns =['Name', 'Seq']) 

l=[]
m=[]
dic={'A':1,'C':2,'G':3,'T':4}

print(dic['A'])
for i in df['Seq']:
    for j in i:
#        print(j)
        if j == 'A':
            l.append(dic['A'])
        if j == 'G':
            l.append(dic['G'])
        if j == 'C':
            l.append(dic['C'])
        if j == 'T':
            l.append(dic['T'])
    m.append(l)
print(len(m)) 

import numpy as np
z = np.array(m)
print(z)
print("z[1] =", z[1])
###########
def hamming_distance(string1, string2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance

example_dist = hamming_distance("GATTACA", "GACTATA")
print(example_dist)

hdv=[]
n=[]
print('%%%%%%%%%%%%%%%')
for s1 in ds:
    for s2 in ds:
        hdv.append(hamming_distance(s1,s2))
#    print(hdv)
    print(len(hdv))
    n.append(hdv)
print(len(n))
################
print('MDSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
n_samples = 120

similarities = euclidean_distances(z)
print(similarities)
# Add noise to the similarities
noise = np.random.rand(n_samples, n_samples)
noise = noise + noise.T
noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
similarities += noise
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=None,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

#nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                    dissimilarity="precomputed", random_state=None, n_jobs=1,
#                    n_init=1)
#npos = nmds.fit_transform(similarities, init=pos)

# Rescale the data
pos *= np.sqrt((z ** 2).sum()) / np.sqrt((pos ** 2).sum())
#npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

# Rotate the data
clf = PCA(n_components=2)
z = clf.fit_transform(z)
print(z)
pos = clf.fit_transform(pos)

#npos = clf.fit_transform(npos)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s = 100
plt.scatter(z[:, 0], z[:, 1], color='navy', s=s, lw=0,
            label='True Position')
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
#plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)
plt.show()
###########
print('PCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
n_samples = 120

similarities = euclidean_distances(z)
print(similarities)
# Add noise to the similarities
noise = np.random.rand(n_samples, n_samples)
noise = noise + noise.T
noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
similarities += noise
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=None,
                   dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_

#nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                    dissimilarity="precomputed", random_state=None, n_jobs=1,
#                    n_init=1)
#npos = nmds.fit_transform(similarities, init=pos)

# Rescale the data
pos *= np.sqrt((z ** 2).sum()) / np.sqrt((pos ** 2).sum())
#npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

# Rotate the data
clf = PCA(n_components=2)
z = clf.fit_transform(z)
print(z)
pos = clf.fit_transform(pos)

#npos = clf.fit_transform(npos)

fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

s = 100
plt.scatter(z[:, 0], z[:, 1], color='navy', s=s, lw=0,
            label='True Position')
plt.scatter(pos[:, 0], pos[:, 1], color='turquoise', s=s, lw=0, label='MDS')
#plt.scatter(npos[:, 0], npos[:, 1], color='darkorange', s=s, lw=0, label='NMDS')
plt.legend(scatterpoints=1, loc='best', shadow=False)

#similarities = similarities.max() / similarities * 100
#similarities[np.isinf(similarities)] = 0
#
## Plot the edges
#start_idx, end_idx = np.where(pos)
## a sequence of (*line0*, *line1*, *line2*), where::
##            linen = (x0, y0), (x1, y1), ... (xm, ym)
#segments = [[X_true[i, :], X_true[j, :]]
#            for i in range(len(pos)) for j in range(len(pos))]
#values = np.abs(similarities)
#lc = LineCollection(segments,
#                    zorder=0, cmap=plt.cm.Blues,
#                    norm=plt.Normalize(0, values.max()))
#lc.set_array(similarities.flatten())
#lc.set_linewidths(np.full(len(segments), 0.5))
#ax.add_collection(lc)

plt.show()
