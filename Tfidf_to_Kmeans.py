# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
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
import multiprocessing
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
stemmer = SnowballStemmer("english")
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
import sklearn


df.info()

os.chdir("/Users/parkerglenn/Desktop/DataScienceSets/Cleaned_Articles")

data = codecs.open('all_GOOD_articles.csv', encoding = 'utf-8')
data_with_labels = codecs.open("/Users/parkerglenn/Desktop/DataScienceSets/Cleaned_Articles/Google_Drive/Article_Classification12.csv")

df = pd.read_csv(data)
labels_df= pd.read_csv(data_with_labels)

for col in df:
    print ("column", col, ":", type(df[col][0]))

df = df.drop(df.columns[:12], axis = 1)

new_df = df[5000:6000]

corpus = []
for text in new_df['content']:
    corpus.append(text)

titles = []
for title in new_df["title"]:
    titles.append(str(title))

events = []
for event in labels_df["Event"][:1000]:
    events.append(str(event))





#############################
#I don't actually end up using this
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.lower().split()
    for word in words:
        word = stemmer.stem(word)
    return words

english_stopwords = stopwords.words('english')
#############################

#TFIDF

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in corpus:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

tf = TfidfVectorizer(analyzer='word', use_idf=True, stop_words = "english", tokenizer = tokenize_and_stem,min_df = 0)

tfidf_matrix =  tf.fit_transform(corpus)
print(tfidf_matrix.shape)
terms = tf.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

##############################################

#KMeans

from sklearn.cluster import KMeans
num_clusters = 500 
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)

#Saves the model you just made
#joblib.dump(km, '500_No_Ngram.pkl')

from sklearn.externals import joblib
#Loads my pre-existing kmeans model
km = joblib.load('/Users/parkerglenn/Desktop/DataScienceSets/Cleaned_Articles/KMeans_Cluster_Models/500_No_Ngram.pkl')
clusters = km.labels_.tolist()

articles = {"title": titles, "date": new_df["date"], "cluster": clusters, "content": new_df["content"], "event": events[:1000]}
frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'date', 'cluster', 'content', "event"])
frame['cluster'].value_counts()

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 


#Creates a count dict (success) to see how many instances of the same event are clustered together
for i in clusters[0:50]:
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print()
    success = {}
    for event in frame.ix[i]["event"].values.tolist():
        if event in success.keys():
            success[event] += 1
        elif event not in success.keys():
            num = 1
            success[event] = num
    print(success)
    print()
    print()

#Allows you to zoom in on a specific cluster, see what words make that cluster unique
for i in clusters:
    if i == 2: #Change 2 to the cluster 
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :20]: #replace 20 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    
    
#This is bad
#Don't use this
def Success_Measure(x):   
    everything = []
    tot_success = []
    for i in x:
        success = {}
        for event in frame.loc[i]["event"].values.tolist():
            if event not in everything and event != "nan" and event != "useless":
                everything.append(event)
            if event in success.keys():
                success[event] += 1
            if event not in success.keys():
                num = 1
                success[event] = num
        tot_success.append(success)
    num = 0
    for cluster in tot_success:
        for key in cluster:
            if key != "nan" and key != "useless":
                num += 1
    out = (len(everything) / num) * 100
    print()
    print()
    print("Success with this model is " + str(out) + " percent")
    print()
    print()

