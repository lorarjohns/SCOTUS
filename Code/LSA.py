#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 00:26:22 2019

@author: lorajohns
"""
import sklearn

tm =  TopicModeler(lemmatize=False, top_words=20)
topics.train = tm.fit_transform(data)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(xtrain, ytrain)
clf.score(xtest, ytest)

from sklearn import metrics
print(metrics.f1_score(clf.predict(xtest), ytest))



# create the count matrix

from numpy import zeros
from scipy.linalg import svd

data = 

# define LSA class

class LSAObject:
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords 
        self.ignorechars = ignorechars 
        self.wdict = {} 
        self.dcount = 0

# parse documents
        
def parse(self, doc):
    words = doc.split(); 
    for w in words:
        w = w.lower().translate(None, self.ignorechars) 
        if w in self.stopwords:
            continue
        elif w in self.wdict:
            self.wdict[w].append(self.dcount)
        else:
            self.wdict[w] = [self.dcount]
            self.dcount += 1
            
# build count matrix
            
def printA(self):
print self.A

# test LSA class ; pass it the stopwords and ignores, call parse
# call build 

mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)
    mylsa.build()
    mylsa.printA()
    
# OPTION - modify with tf-idf
    
# SVD
    
def calc(self):
    self.U, self.S, self.Vt = svd(self.A)
    
# assumes gaussian distribution and frobenius norm
    
    from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(12,8))
scores = list(zip(tfidf.get_feature_names(), chi2score))
chi2 = sorted(scores, key=lambda x:x[1])
topchi2 = list(zip(*chi2[-20:]))
x = range(len(topchi2[1]))
labels = topchi2[0]
plt.barh(x,topchi2[1], align='center', alpha=0.5)
plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)
plt.yticks(x, labels)
plt.xlabel('$\chi^2$')
plt.show();
We can observe that the features with a high χ2 can be considered relevant for the sentiment classes we are analyzing.