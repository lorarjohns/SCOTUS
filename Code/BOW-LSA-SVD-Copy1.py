#!/usr/bin/env python
# coding: utf-8

# In[46]:


# Import tfidf and normalizer for bag of words and pre-processing
# Import LDA, K-means clustering, NMF, and LSA 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# Import general use tools
import pandas as pd
import numpy as np
import time

# Import visualization libraries
import matplotlib.pyplot as plt
import matplotlib
#import plotly.express as px
#import plotly


# In[47]:


# Import stop_words and create list to hold additions
from sklearn.feature_extraction import text 

my_additional_stop_words=['15','2628','446','2607','419''____','petition','supreme','rehearing','sugg','plaintiff',
                          'error','employés', '000','ch','said','company','united', 'federal', 'district', 'right',
                          'id', 'opinion','law', 'case', 'state', 'court','sentence','petitioner','pub','280','ch',
                          'statute','case','ct', 'mr', 'ถถ', 'งง', 'zzz','supra','infra','appellant','appellee', 'id',
                          '413', '93', '37','1973','act', 'make', 'ante', 'cite', 'claim', 'respondent','rule','shall',
                          'judgment','say', 'ed', '2d', 'ct']

# Update the in-built stopwords list
stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


# In[51]:


df = pd.read_csv('14th_cleaned_corpora.csv')


# In[52]:


df.sort_values(by="year", inplace=True)


# In[53]:


# one missing year
df.at[1807, 'year'] = 2019


# In[54]:


df


# ### Create a Bag of Words with TF-IDF
# 
# Get columns required for analysis

# In[55]:


cols = ['id_x', 'year', 'case_name', 'corpora']


# In[56]:


corp = df[cols].copy()


# In[57]:


corp


# In[36]:


# fit method creates bag of words. See them with .get_feature_names()
# This returns a sparse matrix. For a dense matrix, you could perform:
# pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names())

# Create a bag of words using ngrams up to 3 words long

vect = CountVectorizer(ngram_range=(1,3))


# In[38]:


# Use .transform() to create a matrix populated with token counts that represent the documents in sparse format.

# def wm_to_df(wm, feat_names):
#    # create an index for each row
#    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
#    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
#                      columns=feat_names)
#    return(df)
#

def make_bow(df_col):
    #custom_vec = CountVectorizer(ngram_range=(1,3))
    custom_vec = CountVectorizer(strip_accents="unicode", stop_words=stop_words)
    corpora = df_col.values.flatten().tolist()
    wm = custom_vec.fit_transform(corpora)
    vocab = custom_vec.vocabulary_
    tokens = custom_vec.get_feature_names()
    
    return wm, tokens, vocab

# Tune the vectorizer with gridsearch...
# ngram_range=(1, 2) # number of ngrams
# max_df=0.75 # ignore terms appearing in >75% of documents 
# min_df=1 # ignore terms that only appear in x documents (dangerous because of case names, which may be unique)


# In[39]:


# Create the matrix and get the features and vocab on the whole corpus
wordmatrix, features, vocab = make_bow(corp['corpora'])


# In[40]:


# Create a dataframe of bag of words vocabulary
def bow_df(vocab):
    vocab_values = list(vocab.values())
    vocab_keys = list(vocab.keys())
    count_df = pd.DataFrame(list(zip(vocab_keys,vocab_values)))
    count_df.columns = ['Word', 'Count']
    count_df.sort_values(by='Count', ascending=False, inplace=True)
    return count_df

count_df = bow_df(vocab)


# ## Perform Latent Semantic Analysis on the corpus as a whole


# Create a pipeline to perform LSA
# Create a vectorizer to convert raw documents to TF/IDF matrix
vectorizer = TfidfVectorizer(stop_words=stop_words,
                             strip_accents="unicode",
                             use_idf=True, 
                             smooth_idf=True)

# This normalizes the vector (L2 norm of 1.0) to neutralize 
# the effect of document length on tf-idf.

normalizer = Normalizer(copy=False)

# Perform singular value decomposition:
# Project the tfidf vectors onto the first N principal components.

svd_model = TruncatedSVD(n_components=100,         # number of dimensions
                         algorithm='randomized',
                         n_iter=10)

lsa_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model),
                            ('norm', normalizer)])



def lsa_transform(df_col):
    corpora = df_col.values.flatten().tolist()
    lsa_matrix = lsa_transformer.fit_transform(corpora)

    print(f"tf-idf params: {lsa_transformer.steps[0][1].get_params()}")

    # Get the words that correspond to each of the features.
    feat_names = lsa_transformer.steps[0][1].get_feature_names()
    vocab = lsa_transformer.steps[0][1].vocabulary_
    
    return lsa_matrix, feat_names, vocab

lsa_matrix, feat_names, lsa_vocab = lsa_transform(corp['corpora'])


# Plot the top 10 terms for each top-10 LSA component. 
for component_num in range(0, 10): # i.e., the top 10 components.

    comp = lsa_transformer.steps[1][1].components_[component_num]
    
    # Sort the weights in the first component and get indices
    indices = np.argsort(comp).tolist()
    
    # Reverse order (largest weights first)
    indices.reverse()
    
    # Get top 10 terms for each component        
    terms = [feat_names[weight_index] for weight_index in indices[0:10]]    
    weights = [comp[weight_index] for weight_index in indices[0:10]]    
   
    # Display these terms and their weights as a horizontal bar graph.    
    # The horizontal bar graph displays the first item on the bottom; reverse
    # the order of the terms so the biggest one is on top.
    terms.reverse()
    weights.reverse()
    positions = np.arange(10) + .5    # Center the bar on the y axis.
    
    plt.figure(component_num)
    plt.barh(positions, weights, align="center")
    plt.yticks(positions, terms)
    plt.xlabel("Weight")
    plt.title(f"Strongest terms for component {component_num+1}")
    plt.grid(True)
    plt.savefig(f"terms_for_component_{component_num+1}")
    plt.show()


# ----------------------------------------------------------------------------
# PREPARE DATA FOR ROLLING WINDOW MODELING
# ----------------------------------------------------------------------------

# Create year ranges and bin the data accordingly.
# To track evolution over time, the cases will be binned
# in a rolling fashion, with overlap, to smooth out
# the effects of the groupings. 

first_year = 1875
last_year  = 2018
increment  = 20
overlap    = increment // 2
# ------------------------

# Include left, exclude right; half-closed, half-open interval [a, b)
# cf. pandas rolling() function

def build_year_ranges(first, last, inc, over):
    year_ranges = []
    for n in range(first, last, over):
        year_ranges.append((n, n + inc))
    return year_ranges

# warning: years must have the same index as data
def put_data_under_year_ranges(data, years, year_ranges):

    # assert len(data) == len(years), \
    # "get_content_under_ranges: data and years do not match length"

    # build a dict with keys = year_ranges, with a list for each range
    data_ranges = dict()
    for y in year_ranges:
        data_ranges[y] = []

    # bin all the data by range - each row should fall in two bins, 
    # if ranges are cleanly overlapped

    # if data is a list
    for i in range(len(data)):
        for y in year_ranges:
            if y[0] <= years[i] and years[i] < y[1]:
                data_ranges[y].append(data[i])
                # this should happen twice for every entry except 
                # the very oldest and the very newest

    return data_ranges

# ------------
# if __name__ == "__main__":
    
cases = []
years = []
corpora = []
              
# bins = build_year_ranges(first_year, last_year, increment, overlap)
# binned_data = put_data_under_year_ranges(cases, years, bins)

# Convert DataFrame columns to lists (warning: keep indices aligned, 
# and make sure cases are sorted by year)

corp_list = corp["corpora"].values.flatten().tolist()
year_list = corp["year"].values.flatten().tolist()
names_list = corp["case_name"].values.flatten().tolist()

# Create a numpy array for all cases 
# in the format array[name][text]

case_dict = np.array(list(zip(names_list, corp_list)))


year_ranges = build_year_ranges(first_year, last_year, increment, overlap)
# binned_data = put_data_under_year_ranges(corp_list, year_list, year_ranges
binned_data = put_data_under_year_ranges(case_dict, year_list, year_ranges)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# PERFORM LATENT SEMANTIC ANALYSIS
# ----------------------------------------------------------------------------
def LSA_per_bin(corpora):

    # assert: first pipeline component must be tfidf/count vectorizer
    lsa_matrix = lsa_transformer.fit_transform(corpora)

    # Get the words that correspond to each of the features.
    feat_names = lsa_transformer.steps[0][1].get_feature_names()
    vocab = lsa_transformer.steps[0][1].vocabulary_

    for component_num in range(0, 10):
    
        comp = lsa_transformer.steps[1][1].components_[component_num]
        
        # Sort the weights in the first component and get indices
        indices = np.argsort(comp).tolist()
        
        # Reverse order (largest weights first)
        indices.reverse()
        
        # Get top 10 terms for component        
        terms = [feat_names[weight_index] for weight_index in indices[0:10]]    
        weights = [comp[weight_index] for weight_index in indices[0:10]] 
        terms.reverse()
        weights.reverse()
       
        # Display these terms and their weights as a horizontal bar graph.    

        bin_terms = terms
        bin_weights = weights
        bin_matrix = lsa_matrix
        bin_feat_names = feat_names
        bin_vocab = vocab
        
    return {"terms": bin_terms, "weights": bin_weights, "matrix": bin_matrix, "feat_names": bin_feat_names, "vocab": bin_vocab}
        
def run_LSA_on_bins(binned_data, year_ranges):    
    
    model_ranges = dict()
    for y in year_ranges:
        model_ranges[y] = []
    
    for y in year_ranges:
        model_ranges[y].append(LSA_per_bin(binned_data[y]))
        print(f"Running cases from: {y}") 
        
    return model_ranges    

lsa = run_LSA_on_bins(binned_data, year_ranges)


for k,v in lsa.items():
    print(f"terms for {k}:\n{lsa[k][0]['terms']}")

# def kmeans_topics(binned_data, n_clusters):
#     vectorizer = TfidfVectorizer(stop_words=stop_words,
#                              strip_accents="unicode",
#                              use_idf=True, 
#                              smooth_idf=True)
#     kmeans = KMeans(n_clusters).fit(vectorizer)
#     kmeans.predict(tfidf_vectorizer.transform(binned_data))
#     
#     #kmeans_pipe = Pipeline([('tfidf', vectorizer), 
#                             #('kmeans', kmeans)]) # ('norm', normalizer)   


# ----------------------------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------------------------
# Create a dataframe to hold the number of opinions
# per year (approximate) and visualize the number
    
for k,v in binned_data.items():
    print(f"Number of cases decided circa {k}: {len(v)}")

years_k = list(year_ranges.keys())
values_v = [len(v) for v in year_ranges.values()]
num_cases = list(zip(years_k, values_v))


num_df = pd.DataFrame(num_cases)
num_df.columns = ["Years", "Approx. No. Opinions Issued"]
num_df.Years = num_df.Years.apply(lambda x: (x[0]+x[1])/2)
#fig = px.bar(num_df, x='Years', y='Approx. No. Opinions Issued')
#fig.show()

# ----------------------------------------------------------------------------
# What is the optimal number of clusters for whole 14th corpus?
# Check using silhouette score.
# ----------------------------------------------------------------------------

from tqdm import tqdm_notebook
distorsions = []
sil_scores = []
k_max = 25
vectorizer = TfidfVectorizer(max_df=0.95,
                                     min_df=1, ngram_range=(1,2),
                                     stop_words=stop_words,
                                     use_idf=True)

#vz = vectorizer.fit_transform(corpora)

for k in tqdm_notebook(range(2, k_max)):
    kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42,  
                         init_size=500, verbose=True, max_iter=1000)
    #kmeans_model.fit(vz)
    
    km_transformer = Pipeline([('tfidf', vectorizer), 
                            ('km', kmeans_model),
                            ('norm', normalizer)])
    
    km = km_transformer.fit_transform(corpora)
    
    # sil_score = silhouette_score(vz, kmeans_model.labels_)
    sil_score = silhouette_score(km, km_transformer.steps[1][1].labels_)
    sil_scores.append(sil_score)
    # distorsions.append(kmeans_model.inertia_)
    distorsions.append(km_transformer.steps[1][1].inertia_)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))

ax1.plot(range(2, k_max), distorsions)
ax1.set_title('Distorsion vs num of clusters')
ax1.grid(True)

ax2.plot(range(2, k_max), sil_scores)
ax2.set_title('Silhouette score vs num of clusters')
ax2.grid(True)


# ----------------------------------------------------------------------------
# DEFINE KMEANS ALGORITHM
# ----------------------------------------------------------------------------

def k_means_cluster(corpora, n_clusters=15, ngrams=(1,1)):
    print("%d documents" % len(corpora))
    
    print("Extracting features from the training dataset "
          "using a sparse vectorizer")
    t0 = time()
    
    vectorizer = TfidfVectorizer(max_df=0.95,
                                     min_df=1, ngram_range=ngrams,
                                     stop_words=stop_words,
                                     use_idf=True)
    km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=100,
                             init_size=500, batch_size=1000)
    X = vectorizer.fit_transform(corpora)
    
    
    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()
    
    
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()
    
    
    print("Silhouette Coefficient: %0.3f"
          % silhouette_score(X, km.labels_)) # sample size=5000
    
    print()
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    
    terms = vectorizer.get_feature_names()
    for i in range(n_clusters):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
            print()       
    
    
    clusters = {}
    for i in range(n_clusters):
        clusters[i] = []
        for ind in order_centroids[i, :10]:
            clusters[i].append(terms[ind])
    clusters['labels'] = km.labels_
    clusters['vocabulary'] = vectorizer.vocabulary_
    return clusters

# ----------------------------------------------------------------------------
# if __name__ == "__main__":
# ----------------------------------------------------------------------------
    
# Do the clustering over time
cluster = k_means_cluster(corpora)
# Create a DataFrame column to match cluster topic labels to documents
corp['topics'] = cluster['labels']
top_cols = ['case_name', 'year', 'topics']
topic_df = corp[top_cols]


model_ranges =  dict() # np.array() # Try this as an array
for y in year_ranges:
    model_ranges[y] = []
    
for y in year_ranges:
    print(f"Running cases from: {y}")
    print()
    model_ranges[y].append(k_means_cluster(binned_data[y][1], n_clusters=6, ngrams=(1,2)))
    # note: index accounts for binned_data as numpy array, not dict


# for k in model_ranges.keys():
#     print(sorted(model_ranges[k][0][0])) #topic 0

# Check to see where terms overlap and how they evolve
    
for k in model_ranges.keys():
    for j in range(len(model_ranges[k][0])-1):
        if "company" in model_ranges[k][0][j]:
            print(f"'company' in range {k}, topic {j}: \n{model_ranges[k][0][j]}\n\n") 
        #for term in model_ranges[k][0][j]:
            #if term in (sorted(model_ranges[k][0][j])):
                #print(f"'{term}' in range {k}, topic {j}: \n{model_ranges[k][0][j]}\n\n")



