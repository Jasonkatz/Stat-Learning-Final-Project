
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt, matplotlib.cm as cm
import codecs, re, pandas as pd, numpy as np, json
from time import time

# Data
from sklearn.datasets import fetch_20newsgroups

# NLP-Tools
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.regexp import RegexpStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

# Classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Input data
stopword_file = '/afs/ee.cooper.edu/user/g/i/gitzel/masters/lda_repos/lda-textmine/python/stoplist.txt'
stemming = True
minlength = 3

categories = ['comp.graphics', 'sci.space', 'soc.religion.christian']

# Model parameters
n_features = 1000
n_topics = len(categories)
# n_topics = 20
n_top_words = 10
n_lda_iter = 10

class Document(object):
    def __init__(self, raw, target):
        self._raw_text = raw
        self._target = target
        self._clean_text = ''
        self._features = []
        self._topic_names = []
        self._topic = []     
   
    @property
    def raw_text(self):
        return self._raw_text
    @raw_text.setter
    def raw_text(self, value):
        self._raw = value
    @property
    def clean_text(self):
        return self._clean_text
    @clean_text.setter
    def clean_text(self, value):
        self._clean_text = value
    @property
    def features(self):
        return self._features
    @features.setter
    def features(self, value):
        self._features = value
    @property
    def topic_names(self):
        return self._topic_names
    @topic_names.setter
    def topic_names(self, value):
        self._topic_names = value
    @property
    def topic(self):
        return self._topic
    @topic.setter
    def topic(self, value):
        self._topic = value
    @property
    def target(self):
        return self._target
    @target.setter
    def target(self, value):
        self._target = value

class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Document):
            return obj.__dict__
        return super(DocumentEncoder, self).default(obj)

# Helper Functions
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


# In[2]:

# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.

print("Loading dataset...")
t0 = time()
dataset = fetch_20newsgroups(categories=categories, shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
documents = [Document(datum, dataset.target_names[target]) for datum, target in zip(dataset.data, dataset.target)]
print("done in %0.3fs." % (time() - t0))

# Stopwords and Stemming
print("Cleaning dataset...")
t0 = time()
stopwords = load_stopwords(stopword_file)
stemmer = SnowballStemmer('english', ignore_stopwords=True)
stemmer.stopwords = stopwords
exp = "'(?!(d|m|t|ll|ve)\W)|[.,\-_!?:;()0-9@=+^*`~#$%&| \t\n\>\<\"\\\/\[\]{}]+"

for document in documents:
    clean_sample = ''
    for token in re.split(exp, document.raw_text.lower()):
        if not token or token in stopwords:
            continue
        if stemming:
            token = stemmer.stem(token)
        if len(token) < minlength:
            continue
        clean_sample = clean_sample + ' ' + token
    document.clean_text = clean_sample
print("done in %0.3fs." % (time() - t0))


# # TF-IDF / TF Vectors

# In[27]:

# tf (raw term count)
print("Extracting tf features...")
tf_vectorizer = CountVectorizer(
    max_df=0.95, stop_words='english', max_features=n_features)
t0 = time()
tf = tf_vectorizer.fit_transform([doc.clean_text for doc in documents])
print("done in %0.3fs." % (time() - t0))

# Print TF Features
# Print the features of the first document to check parsing output.
feature_map = {feature: word for word, feature in tf_vectorizer.vocabulary_.iteritems()}    
doc_index = 0
print 'Document: {}'.format(documents[doc_index].raw_text)
tf_array = tf.toarray()
for i, feature in enumerate(tf_array[doc_index]):
    if feature > 0.:
        print 'Feature #{}\t{:.4f}\t{}'.format(i, feature, feature_map[i])

# for i, doc in enumerate(documents):
#     doc.features = list(tf_array[i])
# print '{} total features'.format(len(tf_vectorizer.vocabulary_))


# # Fit LDA Model w/ TF

# In[4]:

print("Fitting LDA model with %d tf features and %d topics..." % (n_features, n_topics))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=n_lda_iter,
                                learning_method='online', learning_offset=20.,
                                random_state=0)
t0 = time()
lda.fit(tf)

topic_word = lda.components_
doc_topic = lda.transform(tf)
print("done in %0.3fs." % (time() - t0))
print("log-likelihood: %0.2f" % lda.score(tf))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
lda_topics = top_words(lda, tf_feature_names, 3)

for i, document in enumerate(documents):
    topic_vector = doc_topic[i]
    document.topic = list(topic_vector / topic_vector.sum())  # normalize distribution
    document.topic_names = lda_topics


# In[7]:

path = '/afs/ee.cooper.edu/user/g/i/gitzel/masters/datasets/newsgroups/3_topics/doc_{:04d}.json'
for i, doc in enumerate(documents):
    with open(path.format(i), 'w') as outfile:
        json.dump(doc, outfile, cls=DocumentEncoder)


# In[6]:

# Join target and predicted topic matrices
target = pd.DataFrame(dataset.target, columns=['target topic'])  # target matrix

df_y_hat = pd.DataFrame(doc_topic, columns=lda_topics)  # document-topic matrix
joined = target.join(df_y_hat)  # join target values
joined.groupby('target topic').mean()  # group by target value, averaging the topic weightings

# Predict generated topic
lr = LogisticRegression()
lr.fit(joined[lda_topics], joined['target topic'])
prediction = lr.predict(joined[lda_topics])

score = accuracy_score(joined['target topic'], prediction) * 100.
matrix = confusion_matrix(joined['target topic'], prediction)

for topic in lda_topics:
    print topic
# print categories
doc = documents[0]
print doc.target
print doc.topic
print doc.raw_text


# In[ ]:




# doc.target, doc.topic, doc.raw_text, doc.clean_text
