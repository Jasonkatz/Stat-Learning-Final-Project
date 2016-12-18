
# coding: utf-8

# # Imports

# In[3]:

#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from time import time
import codecs, re, pandas as pd
import numpy as np
from Tkinter import *

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

from sklearn.manifold import TSNE


# # Setup

# In[4]:

stemming = True
minlength = 3
stopword_file = 'stoplist.txt'

n_features = 100
n_topics = 3
n_top_words = 10


# # Helper Functions

# In[5]:

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

# In[6]:

import os
folder = 'tweets/all/'
data_samples = []
for filename in os.listdir(folder):
    with open(folder + filename) as infile:
        data_samples.append(infile.read())


# # Stopwords and Stemming

# In[7]:

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

# In[8]:

# tf (raw term count)
print("Extracting tf features...")
tf_vectorizer =CountVectorizer(max_df=0.95, stop_words='english',
                               max_features=n_features)
t0 = time()
tf = tf_vectorizer.fit_transform(clean_data_samples)
print("done in %0.3fs." % (time() - t0))


# # Fit LDA Model w/ TF

# In[37]:

n_topics = 3
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


# In[35]:

print lda_topics
