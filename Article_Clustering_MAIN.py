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

os.chdir("/Users/parkerglenn/Desktop/DataScience/Article_Clustering")

data = codecs.open('/Users/parkerglenn/Desktop/DataScience/Article_Clustering/csv/all_GOOD_articles.csv', encoding = 'utf-8')
data_with_labels = codecs.open("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/Google_Drive/Article_Classification122.csv")

df = pd.read_csv(data)
labels_df= pd.read_csv(data_with_labels)

for col in df:
    print ("column", col, ":", type(df[col][0]))
#Deletes unnecessary columns
df = df.drop(df.columns[:12], axis = 1)
#Sets manageable range for working data set
new_df = df[5000:6000]
#Gets info in list form to be later called in kmeans part

corpus = []
for text in new_df['content']:
    corpus.append(text)

titles = []
for title in new_df["title"]:
    titles.append(str(title))
#labels_df starts at df[5000] so we're good on the matching of labels to content
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

##############################################################################
##########TFIDF###############################################################
##############################################################################

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

#Let's you search with stemmed word to see original format of word
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

tf = TfidfVectorizer(analyzer='word', use_idf=True, stop_words = "english", tokenizer = tokenize_and_stem,min_df = 0)

tfidf_matrix =  tf.fit_transform(corpus)
print(tfidf_matrix.shape)
terms = tf.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)



##############################################################################
###################KMEANS#####################################################
##############################################################################
from sklearn.externals import joblib
#Loads my pre-existing kmeans model
#Saves the model you just made
#joblib.dump(km, '700_No_Ngram.pkl')
km = joblib.load("/Users/parkerglenn/Desktop/DataScience/Article_Clustering/KMeans_Cluster_Models/500_no_Ngram.pkl")
clusters = km.labels_.tolist()



#Only to create a new kmeans model
from sklearn.cluster import KMeans
num_clusters = 700
km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()


articles = {"title": titles, "date": new_df["date"], "cluster": clusters, "content": new_df["content"], "event": events[:1000]}
frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'date', 'cluster', 'content', "event"])
frame['cluster'].value_counts()

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 


from collections import Counter
#Creates a count dict (success) to see how many instances of the same event are clustered together
for i in clusters[:100]:
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print()
    counts = []
    for event in frame.loc[i]["event"].values.tolist():
        counts.append(event)
    counts = dict(Counter(counts))    
    print(counts)
    print()
    print()


#Allows you to zoom in on a specific cluster, see what words make that cluster unique
for i in clusters:
    if i == 113: #Change 2 to the cluster 
        print("Cluster %d words:" % i, end='')
        for ind in order_centroids[i, :20]: #replace 20 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
        counts = []
        for event in frame.ix[113]["event"].values.tolist():
            counts.append(event)
        counts = dict(Counter(counts))    
        print(counts)
        print()
        print()
        
        

##############################################################################
##################SUCESS RATES################################################
##############################################################################


"""
BELOW THIS CREATES DICT OF CLUSTERS AND PREDOMINANT EVENT

If multiple events occur the same amount of times in a single cluster,
the ratio function is invoked to choose the event holding the most relative
significance. If one ratio is not greater than the others (ex. a cluster 
composed of 5 one-off events) then the cluster is disregared (labelled "nan").

If the cluster only contians one event, it is assumed at this stage that it is 
the main cluster for the event. 

BUGS:
    If the cluster contains only "nan" events, it will not show up in y_trueDict
    (ex. Cluster 113 is not shown, consisting of {'nan': 2} )
""" 
from collections import Counter
all_events = []
#This fixes quirk where the same cluster was iterated over multiple times
clusters_we_saw = []
for cluster in clusters: 
    if cluster not in clusters_we_saw:
        clusters_we_saw.append(cluster)
        for event in frame.loc[cluster]["event"].values.tolist():
            if event != "nan" and event != "useless":
                all_events.append(event)
event_occurences = dict(Counter(all_events))

y_trueDict = {}
#This range needs to be changed depending on the cluster model
for i in range(0,500):
    ratios = []
    counts = []
    cluster_event = []
    ratio = 0
    
    #Counts occurence per cluster of event
    for event in frame.loc[i]["event"].values.tolist():
        if event != "nan" and event != "useless":
            counts.append(event)
    counts = Counter(counts) 
    
    
    if len(counts) > 1:
        score_1 = list(counts.most_common()[0])[1]
        score_2 = list(counts.most_common()[1])[1]
        #Check to see if there are multiple events with same frequency
        if score_1 == score_2:
            #Gets all events with same frequency
            tied_events = [k for k,v in dict(counts).items() if v == score_1]
            for event in tied_events:
                #Gets the ratio of an occurence for an event in a cluster
                #For example, if an event happens only once, it's ratio will be 1
                #But if "iowa_caucuses" is used 100 times and only 20 times in a specific cluster,
                #its ratio is .2
                new_ratio = score_1 / int(event_occurences[event])
                ratios.append(new_ratio)
                if new_ratio > ratio:
                    cluster_event = event
                    ratio = new_ratio
                #If result is an empty list, all ratios are unique. If not, there
                #are repititions and the data point is thrown out.
                if list(set([x for x in ratios if ratios.count(x) > 1])) != []:
                     y_trueDict[i] = "nan"
                     break
                
                #Dumb try and except sees if ytrueDict[i] is already set to something ("nan")
                try: 
                    y_trueDict[i] 
                except:
                    counts = dict(counts)
                    #Makes sure there's still the occurence in cluster attached to the cluster_event
                    y_trueDict[i] = [cluster_event, counts[cluster_event]]
    

    #If there is one obviously right event, i.e. score_1 != score_2
        else:
            y_trueDict[i] = list(counts.most_common()[0])
    
    #Catches the instance of only one item per cluster, i.e. len(counts) !> 1
    elif len(counts) == 1:
        y_trueDict[i] =list(counts.most_common()[0]) 


#Re-analyzes y_trueDict, applying ratio again so there's one objectively "right" cluster per event
a = []
for k in y_trueDict:
    a.append(y_trueDict[k][0])
a = dict(Counter(a))

#Sees where the same event label is applied to multiple clusters
duplicates = []
for g in a:
    if a[g] > 1 and g != "n" and g != "unknown":
        duplicates.append(g)


#Creates dup_eventsPLUSratio, where the duplicate events are stored by cluster number
#with their ratio
dup_eventsPLUSratio = {}
for key in y_trueDict:
    if y_trueDict[key][0] in duplicates:
        event = y_trueDict[key][0]
        ratio = int(y_trueDict[key][1]) / int(event_occurences[event])
        eventPLUSratio = []
        eventPLUSratio.append(event)
        eventPLUSratio.append(ratio)
        dup_eventsPLUSratio[key] = eventPLUSratio
dup_eventsPLUSratio

#Dives into dup_eventsPLUSratio to see what cluster is more approrpiate for event
for duplicate in duplicates:
    ratios = []
    for key in dup_eventsPLUSratio:
        if dup_eventsPLUSratio[key][0] == duplicate:
            ratios.append(dup_eventsPLUSratio[key][1])
    sort=sorted(ratios,reverse=True)
    highest = sort[0]
    theGood_one = [duplicate, highest]
    for key in dup_eventsPLUSratio:
        if dup_eventsPLUSratio[key][0] == duplicate:    
            if dup_eventsPLUSratio[key] != theGood_one or highest == sort[1]:
                y_trueDict[key] = "nan"
        #If after all that there's still a tie between the top two ratios,
        #(like in hail_caesar_movie where its split 2 and 2 between clusters)
        #its given a "nan" label
        #COULD BE CHANGED TO FIT A WHILE LOOP THAT THEN FINDS score_2 AND
        #RELABELS CLUSTER TO SECOND MOST POPULAR EVENT IF THAT EVENT IS NOT
        #ALREADY ASSIGNED A CLUSTER

#Gets y_true, the correct cluster assignments for each event
bad_labels = ["useless","nan","unkown"]
y_true = []          
for event in events[:1000]:
    find = False
    for key in y_trueDict:
        #Used to see if there is a distinct cluster for that event
        #FIXED BUG: probably still some duplicates in y_trueDict somehow, bc output len is 10005
        #maybe the "unknown" or "useless" stuff?
        if y_trueDict[key][0] == event and y_trueDict[key][0] != "useless" and y_trueDict[key][0] != "nan" and y_trueDict[key][0] != "unknown":
            y_true.append(key)
            find = True
    if find == False:
        #Arbitrary value that's not going to return a match in t score
        y_true.append("nan")


#Gets y_pred, the cluster where each individual event was actually clustered
y_pred = []
for cluster_assignment in frame["cluster"]:
    y_pred.append(cluster_assignment)

#checks how events actually match up with definitively defined cluster
num = 0
for i in y_true:
    if i != "nan":
        num += 1
num

#Re-Aligns two lists to only include good values (those not equalling "nan")
filtered_y_true = []
filtered_y_pred = []   

for place in range(len(y_true)):
    if y_true[place] != "nan":
        filtered_y_true.append(y_true[place])
        filtered_y_pred.append(y_pred[place])


from sklearn.metrics import f1_score
f1_score(y_true = filtered_y_true, y_pred = filtered_y_pred, average = "micro")

#500_no_ngrams F1 score micro: 0.8785046728971962 (also works off the most samples)
#350_3_ngrams F1 score micro: 0.8718861209964412 (but goes off 281 samples rather than 303 in no ngrams)
#700_no_ngrams F1 score micro: 0.8638392857142858
#350_no_ngrams F1 score micro: 0.8576158940397351
#300_3_ngrams F1 score micro: 0.8294573643410853

#Silhouette values lies in the range of [-1, 1]. A value of +1 indicates that the sample is far away 
#from its neighboring cluster and very close to the cluster its assigned. Similarly, value of -1 
#indicates that the point is close to its neighboring cluster than to the cluster its assigned. 
#And, a value of 0 means its at the boundary of the distance between the two cluster. Value of +1 
#is ideal and -1 is least preferred. Hence, higher the value better is the cluster configuration.

from sklearn.metrics import silhouette_score
silhouette_score(tfidf_matrix, y_pred)

#500_no_ngrams: 0.07096239881264323
#350_no_ngrams: 0.06777628195061947
#700_no_ngrams: 0.06251251395097632
#350_3_ngrams: 0.04969413068018369
#300_3_ngrams: 0.04857286650243616

###############################################################################
##################VISUALIZATION################################################
###############################################################################
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import scipy.sparse as sps
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot



#Distance Heatmap
cmap = pyplot.cm.cubehelix
dimensions = (20,20)
fig, ax = pyplot.subplots(figsize=dimensions)
sns.heatmap(dist, vmin = 0, vmax = 1, cmap = cmap).set_title("Tfidf Distances Between Articles", fontsize = 15)
""" 
Notice hot spot around the 625:630 line. 
Those article titles:
['Dem Debate Blogging #1',
 'Dem Debate Blogging #2',
 'Dem Debate Blogging #3',
 'Dem Debate Blogging #4',
 'Dem Debate Blogging #5',
 'Dem Debate Blogging #6']


Around 90:160: large circle of relatively similar articles.
Reason: Iowa Caucuses.
"""




#Tfidf Matrix, in 2D SVD scatterplot 
svd = TruncatedSVD(n_components=2).fit(tfidf_matrix)
data2D = svd.transform(tfidf_matrix)
plt.title("Truncated SVD, 2 Components")
colors = rng.rand(1000)
plt.scatter(data2D[:,0], data2D[:,1], marker = "o", c = colors, cmap = "YlGnBu", s = 10)
"""
Seems to follow Zipf's law: Rank in corpus * number = constant
"""


######With clusters assigned as colors########
from cycler import cycler
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 350)
kmeans.fit(data2D)
y_kmeans = kmeans.predict(data2D)

"""Zoomed out plot with colors"""
plt.title("Truncated SVD with Colored Clusters")
plt.scatter(data2D[:, 0], data2D[:, 1], c=y_kmeans, cmap = "tab20", s=30)

"""Example plot, zoomed in to visualize cluster centers"""
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=30, alpha=0.5, marker = 'x');
pyplot.axis(ymax = 0, ymin =-.25 , xmax = .2, xmin = .14)
plt.title("Truncated SVD with Cluster Centers")
plt.scatter(data2D[:, 0], data2D[:, 1], c=y_kmeans, cmap = "tab20", s=30)


######Interactive ScatterPlot with SVD######
from sklearn.cluster import KMeans
import mpld3
from mpld3 import display
svd = TruncatedSVD(n_components=2).fit(tfidf_matrix)
data2D = svd.transform(tfidf_matrix)

fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
ax.plot(data2D[:, 0], data2D[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('Truncated SVD with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    print(i)
    plt.annotate(txt + ", " + str(y_pred[i]), (data2D[:, 0][i], data2D[:, 1][i]))
mpld3.show(fig)
mpld3.save_html(fig, "Truncated_SVD_D3.html")




#####Interactive ScatterPlot with Dense Matrix and PCA######
"""Probably not the method to use. SVD seems better since it takes the sparse matrix "tfidf_matrix" directly."""
x = tfidf_matrix.todense()
coords = PCA(n_components=2).fit_transform(x)

fig, ax = plt.subplots(figsize = (14,8))
np.random.seed(0)
ax.plot(coords[:, 0], coords[:, 1],
        'or', ms=10, alpha=0.2)
ax.set_title('PCA with Cluster Assignments', size=14)
ax.grid(color='lightgray', alpha=0.7)
for i, txt in enumerate(events):
    plt.annotate(txt, (coords[:, 0][i], coords[:, 1][i]))
mpld3.show(fig)

    
