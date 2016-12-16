import nltk
import string
import os
import csv
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()

path = 'tweets'

def stem_tokens(tokens, stemmer):
    stemmed = [ ]
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

for subdir, dirs, files in os.walk(path):
    for file in files:
        file_path = subdir + os.path.sep + file
        tweets = open(file_path, 'r')
        text = tweets.read()
        lowers = text.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[file] = no_punctuation
print 'Done preprocessing'

# This can take some time
print 'Performing TF-IDF vectorization'
tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
tfs = tfidf.fit_transform(token_dict.values())

datalist = []
feature_names = tfidf.get_feature_names()
for col in tfs.nonzero()[1]:
    datalist.append([feature_names[col], tfs[0, col]])

# Print data to csv
csvfile = open('tfidf/' + sys.argv[1] + '.csv', 'wb')
wr = csv.writer(csvfile);
wr.writerow(datalist)
wr.close();
