
# coding: utf-8

# In[2]:

import json
from os import listdir
from os.path import isfile, join

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

class DocumentDecoder(json.JSONDecoder):
    def __init__(self, encoding=None, object_hook=None, parse_float=None,
                 parse_int=None, parse_constant=None, strict=True, object_pairs_hook=None):
        json.JSONDecoder.__init__(self, encoding, object_hook, parse_float, parse_int,
                                  parse_constant, strict, object_pairs_hook)

def as_document(d):
    doc = Document(raw=d['_raw_text'], target=d['_target'])
    return doc


# In[11]:

targets = set()
path = '/afs/ee.cooper.edu/user/g/i/gitzel/masters/datasets/newsgroups/3_topics/'
for f in listdir(path):  
    file_path = path + f
    with open(file_path, 'r') as infile:
        d = json.load(infile)
    targets.add(d['_target'])

