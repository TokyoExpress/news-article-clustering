# newsfeed-nlp-processor
In a political climate of intensely polarizing takes on the latest scandals, international relations, and national issues, it can be difficult to make sense of all the data. 
This allows the average news consumer to experience a stream-lined information acquisition process, free of any useless repetition.
We intend to utilize an unsupervised learning algorithim to cluster together a data set of news stories by topic, creating homogeneous groups that provide differing perspective on the same story.
We will be coding in Python, using a NLP software (probably Spacy) to disseminate the large bodies of text and derive coherent and comparable meaning (maybe using word vectors). 


Dataset: https://www.kaggle.com/snapcrack/all-the-news

Relevant Research: http://mkusner.github.io/publications/WMD.pdf
https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html

Possible Roadblocks:
* Specificity in clusters. Russia = Russian election hacking or Russian international relations or Russian Olympic ban?
* Intersections in broad topics. E.g. "Donald Trump speaks about Hurricane Matthew" about DT or hurricane?


Notes:
* Put special emphasis on dates and names when organizing clusters (use spacy entity recognition) 



