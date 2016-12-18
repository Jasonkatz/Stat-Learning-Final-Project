import nltk
import string
import os
import csv
import sys
import decimal

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()

path = 'tweets'

print 'Preprocessing data'
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
print 'TF-IDF vectorization complete'

# Print data to csv
print 'Generating CSV file'
datalist = []
feature_names = tfidf.get_feature_names()
for col in tfs.nonzero()[1]:
    datalist.append([feature_names[col].encode('utf-8'), decimal.Decimal(tfs[0, col])])
if len(sys.argv) > 1:
    filename = 'tfidf/' + sys.argv[1] + '.csv'
else:
    filename = 'tfidf/tweets.csv'
csvfile = open(filename, 'wb')
wr = csv.writer(csvfile);
wr.writerows(datalist)
print 'CSV file generated'
