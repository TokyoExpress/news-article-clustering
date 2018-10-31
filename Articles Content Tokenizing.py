#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import sklearn.manifold
import matplotlib.pyplot as plt
import nltk
import regex as re
import re
import codecs
import csv
import glob
import spacy
nlp = spacy.load("en")
from spacy import displacy
import multiprocessing
import gensim.models.word2vec as w2v
from gensim.similarities import Similarity
from gensim.corpora.textcorpus import TextCorpus
from gensim.similarities import MatrixSimilarity
from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile


# In[3]:


os.chdir("/Users/parkerglenn/Desktop/DataScienceSets/Cleaned_Articles")


# In[4]:


data = codecs.open('all_GOOD_articles.csv', encoding = 'utf-8')


# In[5]:


dataframe_all = pd.read_csv(data)
num_rows = dataframe_all.shape[0]


# In[6]:


counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]


# In[7]:


dataframe_all = dataframe_all[counter_without_nan.keys()]


# In[8]:


for col in dataframe_all:
    print ("column", col, ":", type(dataframe_all[col][0]))


# In[9]:


def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words


# In[10]:


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.corpus import stopwords


# In[11]:


english_stopwords = stopwords.words('english')


# In[12]:


content = dataframe_all['content']


# In[13]:


tokens = []
for x in content:
    words = sentence_to_wordlist(x)
    tokens.append(words)


# In[14]:


filtered_tokens = []
for article in tokens:
    articles = []
    for word in article:
        if word.lower() not in english_stopwords: 
            articles.append(word.lower())
    filtered_tokens.append(articles) 


# In[15]:


len(filtered_tokens)


# In[16]:


df = pd.read_csv('all_GOOD_articles.csv')
new_column = pd.DataFrame({'tokens': [i for i in filtered_tokens]})
df = df.merge(new_column, left_index = True, right_index = True)
df.to_csv('all_GOOD_articles_tokens.csv')


# In[ ]:





# In[ ]:




