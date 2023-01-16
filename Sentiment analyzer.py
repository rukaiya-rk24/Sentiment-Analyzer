#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import nltk
nltk.download('stopwords')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[3]:


train = pd.read_csv(r"C:\Users\MY\Downloads\training_twitter_x_y_train.csv")
test = pd.read_csv(r"C:\Users\MY\Downloads\test_twitter_x_test.csv")


# Cleaning the data

# In[4]:


drop_cols = ['airline_sentiment_gold','name','tweet_id', 'retweet_count','tweet_created','user_timezone','tweet_coord','tweet_location']
train.drop(drop_cols,inplace=True,axis=1)
test.drop(drop_cols,inplace=True,axis=1)
train


# In[4]:


stops = stopwords.words('english')
stops += list(punctuation)
stops += ['flight','airline','flights','AA']
abbreviations = {'ppl': 'people','cust':'customer','serv':'service','mins':'minutes','hrs':'hours','svc': 'service',
           'u':'you','pls':'please'}


# In[5]:


train_index = train[~train.negativereason_gold.isna()].index #Getting all those indexes for which column neg.. is not empty
test_index = test[~test.negativereason_gold.isna()].index


# In[6]:


import re
for index,row in train.iterrows():
    tweet = row.text
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #Remove links
    tweet = re.sub('@[^\s]+','',tweet) #remove usernames
    tweet = re.sub('[\s]+', ' ', tweet) #remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #replace #word with word r stands for raw string
    tweet = tweet.strip('\'"') #trim tweet
    words = []
    for word in tweet.split():
        if word.lower() not in stops:
            if word in list(abbreviations.keys()):
                words.append(abbreviations[word])
            else:
                words.append(word.lower())
    tweet = " ".join(words)
    tweet = " %s %s" % (tweet, row.airline) #in place of %s the value in parenthesis will come
    row.text = tweet
    if index in train_index:
        row.text = " %s %s" % (row.text, row.negativereason_gold)
for index, row in test.iterrows():
    tweet = row.text
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet) #remove links
    tweet = re.sub('@[^\s]+','',tweet) #remove usernames
    tweet = re.sub('[\s]+', ' ', tweet) #remove additional whitespaces
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) #replace #word with word
    tweet = tweet.strip('\'"') #trim tweet
    words = []
    for word in tweet.split(): 
#         if not hasNumbers(word):
        if word.lower() not in stops:
            if word in list(abbreviations.keys()):
                words.append(abbreviations[word])
            else:
                words.append(word.lower())
    tweet = " ".join(words)
    tweet = " %s %s" % (tweet, row.airline)
    row.text = tweet
    if index in test_index:
        row.text = " %s %s" % (row.text, row.negativereason_gold)

del train['negativereason_gold']
del test['negativereason_gold']


# In[7]:


def deemojify(inp):
    return inp.encode('ascii', 'ignore').decode('ascii')


# In[8]:


for index,row in train.iterrows():
    row.text =  deemojify(row.text)
for index,row in test.iterrows():
    row.text =  deemojify(row.text)


# In[9]:


def hasNumber(inp):
    return any(char.isdigit() for char in inp)
for index,row in train.iterrows():
    tweet = row.text
    new_words = []
    for word in tweet.split():
        if not hasNumber(word):
            new_words.append(word)
    row_text = " ".join(new_words)
for index,row in test.iterrows():
    tweet = row.text.split()
    new_words = []
    for word in tweet:
        if not hasNumber(word):
            new_words.append(word)
    row_text = " ".join(new_words)


# In[10]:


train.head()


# In[12]:


v = TfidfVectorizer(analyzer='word',max_features=3150,max_df = 0.8, ngram_range=(1,1))
train_features = v.fit_transform(train.text)
test_features = v.transform(test.text)


# In[16]:


clf = SVC(kernel="linear", C= 0.96 , gamma = 'scale')
# clf = SVC(C = 1000, gamma = 0.001)
clf.fit(train_features, train['airline_sentiment'])
pred = clf.predict(test_features)


# In[17]:


clf = LogisticRegression(C = 2.1, solver='liblinear', multi_class='auto')
clf.fit(train_features,train['airline_sentiment'])
pred = clf.predict(test_features)
with open('predictions_twitter.csv', 'w') as f:
    for item in pred:
        f.write("%s\n" % item)


# In[18]:


with open('predictions_twitter2.csv', 'w') as f: #less accurate
    for item in pred:
        f.write("%s\n" % item)


# In[19]:


v.get_feature_names()


# In[ ]:




