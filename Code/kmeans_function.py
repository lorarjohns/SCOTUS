from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser

import sys
from time import time

import numpy as np
import pandas as pd

df = pd.read_csv('cleaned_corpora.csv')
df.at[1807, 'year'] = 2019
cols = ['id', 'year', 'corpora']
corp = df[cols].copy()
corpora = corp['corpora'].values.flatten().tolist()


def k_means_clustering(n_components=0, n_features=2048576, use_idf=True, n_features=10000, use_hashing=False, verbose=1, use_lsa=True, minibatch=True, n_clusters_minibatch=15):


  print("%d documents" % len(corpora))
  
  print("Extracting features from the training dataset "
        "using a sparse vectorizer")
  
if use_hashing:
    if use_idf:
        # Perform an IDF normalization on the output of HashingVectorizer
        hasher = HashingVectorizer(n_features=n_features,
                                   stop_words=stop_words, alternate_sign=False,
                                   norm=None, binary=False)
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=n_features,
                                       stop_words=stop_words,
                                       alternate_sign=False, norm='l2',
                                       binary=False)
else:
    vectorizer = TfidfVectorizer(max_df=0.85, max_features=n_features,
                                 min_df=1, stop_words=stop_words,
                                 use_idf=use_idf)
X = vectorizer.fit_transform(corpora)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

if n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()

# #############################################################################
# Do the actual clustering

if minibatch:
    km = MiniBatchKMeans(n_clusters=n_clusters_minibatch, init='k-means++', n_init=100,
                         init_size=500, batch_size=1000, verbose=verbose)
else:
    km = KMeans(n_clusters=n_clusters_minibatch, init='k-means++', max_iter=100, n_init=1,
                verbose=opts.verbose)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(km.labels_))
#print("Completeness: %0.3f" % metrics.completeness_score(km.labels_))
#print("V-measure: %0.3f" % metrics.v_measure_score(km.labels_))
#print("Adjusted Rand-Index: %.3f"
#      % metrics.adjusted_rand_score(km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_)) # sample size=5000

print()


if not use_hashing:
    print("Top terms per cluster:")

    if n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(15):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()