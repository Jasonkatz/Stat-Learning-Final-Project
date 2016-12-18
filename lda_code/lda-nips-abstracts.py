
# coding: utf-8

# # Imports

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
import codecs, re, pandas as pd
import numpy as np

# Data
from sklearn.datasets import fetch_rcv1

# NLP-Tools
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# # Setup

# In[2]:

stemming = True
minlength = 3
stopword_file = '/afs/ee.cooper.edu/user/g/i/gitzel/masters/lda_repos/lda-textmine/python/stoplist.txt'

n_features = 10000
n_topics = 2
n_top_words = 10


# # Helper Functions

# In[3]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

def top_words(model, feature_names, n_top_words):
    return [tuple([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) for topic in model.components_]

def load_stopwords(stopword_filename):
    stopwords = set()
    with codecs.open(stopword_filename, 'r', 'utf-8') as sf:
        for line in sf:
            if len(line.split()) != 1:
                print('ignoring line with more than one stopword:\n"{0}"'.format(line))
                continue
            stopwords.add(line.strip())
    return stopwords


# # Load Data

# In[4]:

import csv
with open('/afs/ee.cooper.edu/user/g/i/gitzel/masters/datasets/nips_papers/Papers.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    data_samples = [row['Abstract'] for row in reader]


# # Stopwords and Stemming

# In[5]:

print("Cleaning dataset...")
t0 = time()
stopwords = load_stopwords(stopword_file)
stemmer = SnowballStemmer('english', ignore_stopwords=True)
stemmer.stopwords = stopwords

clean_data_samples = []
for sample in data_samples:
    clean_sample = ''
    for token in re.split("'(?!(d|m|t|ll|ve)\W)|[.,\-_!?:;()0-9@=+^*`~#$%&| \t\n\>\<\"\\\/\[\]{}]+", sample.lower().decode('utf-8')):
        if not token or token in stopwords:
            continue
        if stemming:
            token = stemmer.stem(token)
        if len(token) < minlength:
            continue
        clean_sample = clean_sample + ' ' + token
    clean_data_samples.append(clean_sample)
print("done in %0.3fs." % (time() - t0))


# # TF-IDF / TF Vectors

# In[6]:

# tf (raw term count)
print("Extracting tf features...")
tf_vectorizer =CountVectorizer(max_df=0.95, stop_words='english',
                               max_features=n_features)
t0 = time()
tf = tf_vectorizer.fit_transform(clean_data_samples)
print("done in %0.3fs." % (time() - t0))


# ## Print TF Features

# feature_map = {feature: word for word, feature in tf_vectorizer.vocabulary_.iteritems()}    
# 
# print 'Document: {}'.format(clean_data_samples[document])
# for i, feature in enumerate(tf.toarray()[document]):
#     if feature > 0:
#         print 'Feature #{}\t{:.4f}\t{}'.format(i, feature, feature_map[i])
# print '{} total features'.format(len(tf_vectorizer.vocabulary_))

# # Fit LDA Model w/ TF

# In[29]:

n_topics=10
print("Fitting LDA models with %d topics and %d tf features..." % (n_topics, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf)

topic_word = lda.components_
doc_topic = lda.transform(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
lda_topics = top_words(lda, tf_feature_names, 3)


# In[30]:

print lda_topics
# print categories
# print lda.perplexity()


# ### TSNE Feature Reduction (for scatter-plot)

# In[31]:

topic_hist = []
for row in doc_topic:
    topic_hist.append(row.argmax())

print sum(topic_hist) / float(len(topic_hist))
    
min_hist = []
for row in doc_topic:
    min_hist.append(row.argmin())
    
print sum(min_hist) / float(len(min_hist))
    
fig = plt.figure()
fig.clf()

plt.hist(topic_hist, bins=range(0, n_topics+1))
plt.show()

fig = plt.figure()
fig.clf()

plt.hist(min_hist, bins=range(0, n_topics+1))
plt.show()

from collections import defaultdict

topic_bins = defaultdict(int)
for b in topic_hist:
    topic_bins[b] += 1 
topic_bins


# In[12]:

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=10)
X_tsne = tsne.fit_transform(doc_topic)

color = [doc.argmax() for doc in doc_topic]


# plt.figure(1, figsize=(10,10), dpi=100)
# plt.scatter(X_tsne[:,0], X_tsne[:,1], c=color)
# # fig.c
# plt.show()

# In[16]:

fig = plt.figure(1, figsize=(10,10), dpi=150)
fig.clf()
frame = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=color, cmap=plt.get_cmap('Paired'), edgecolor='', s=50)
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.show()


# import seaborn as sns
# import sklearn.cluster as cluster
# 
# sns.set_context('poster')
# sns.set_color_codes()
# plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
# 
# def plot_clusters(data, algorithm, args, kwds):
#     start_time = time()
#     labels = algorithm(*args, **kwds).fit_predict(data)
#     end_time = time()
#     palette = sns.color_palette('deep', np.unique(labels).max() + 1)
#     colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
#     plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
#     frame = plt.gca()
#     frame.axes.get_xaxis().set_visible(False)
#     frame.axes.get_yaxis().set_visible(False)
#     plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
# #     plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
#     plt.grid()

# In[18]:

import sklearn.cluster as cluster

def plot_clusters(data, algorithm, args, kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    fig = plt.figure(1, figsize=(10,10), dpi=150)
    fig.clf()
    plt.scatter(data.T[0], data.T[1], c=labels, cmap=plt.get_cmap('Set1'), edgecolor='', s=50)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.show()
    
import hdbscan
plot_clusters(X_tsne, hdbscan.HDBSCAN, (), {'min_cluster_size':25})

