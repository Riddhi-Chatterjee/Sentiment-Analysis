#Used to create the training and test datasets in the required format

# %% [markdown]
# # ML Project (Harsh Comment Classification)

# %% [markdown]
# Team Name: Tech Knights
# 
# Team Members: Surya Sastry (IMT2020079), Riddhi Chatterjee (IMT2020094) 

# %% [markdown]
# ## 1. Preprocessing

# %% [markdown]
# ### 1.1 Imports and analysis:

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
train_df = pd.read_csv("harsh-comment-classification/train.csv")
test_df = pd.read_csv("harsh-comment-classification/test.csv")
sample_df = pd.read_csv("harsh-comment-classification/sample.csv")

# %%
train_df.loc[0:20,:]

# %%
len(train_df)

# %%
train_df.nunique()

# %%
train_df.describe()

# %% [markdown]
# We can see that the "id" column is useless. So lets drop the "id" column from train_df...

# %%
train_df.drop(axis="columns", labels="id", inplace=True)
train_df.loc[0:20,:]

# %%
test_df.loc[0:20,:]

# %%
len(test_df)

# %%
test_df.nunique()

# %%
test_df.describe()

# %%
sample_df.loc[0:20,:]

# %%
sample_df.describe()

# %% [markdown]
# ### 1.2 Checking for missing values:

# %%
train_df.isna().sum()

# %% [markdown]
# There are no missing values in the training dataset. Checking further...

# %%
(train_df == "?").sum()

# %% [markdown]
# Indeed there are no missing values in the training dataset

# %%
test_df.isna().sum()

# %% [markdown]
# No missing values in the test dataset. Checking further...

# %%
(test_df == "?").sum()

# %% [markdown]
# There are no missing values in the test dataset for sure...

# %% [markdown]
# ### 1.3 Cleaning the data and performing lemmatization:

# %%
import nltk
nltk.download("all")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
#import contractions

lmt = WordNetLemmatizer()

#unnecessary_words = set(set(stopwords.words("english"))-set(['no', 'nor', 'not', 'against']))

#   NOTE: Initially, 'negative words' were not considered as unnecessary words.
#         But our models are currently giving less ROC_AUC score if this is done.
#         So now, negative words are also considered as unnecessary words.

unnecessary_words = [set(set(stopwords.words("english")))]

def isAllCaps(word):
    if word == word.upper():
        return True
    else:
        return False
    
def transformText(text):
    #text = contractions.fix(text) #Expanding contractions
    
    #   NOTE: Contractions like "isn't", "haven't" etc were being expanded only when we were not considering 'negative words' as unnecessary words.
    #         Currently, it is useless to expand contractions
    
    text = ' '.join(text)
    #text = re.sub('[^a-zA-Z]', ' ', text) #Replacing all characters except a-z and A-Z, by a whitespace character
    text = text.split() #Extracting words from the text
    #text = [word.lower() if not isAllCaps(word) else word.lower() for word in text] #Converting all words to lowercase
    text = [lmt.lemmatize(word) for word in text if not word in unnecessary_words] #Performing lemmatization and removing unnecessary words
    text = ' '.join(text)
    return text

train_df['text'] = train_df['text'].apply(word_tokenize)
train_df["text"] = train_df["text"].apply(transformText)
test_df['text'] = test_df['text'].apply(word_tokenize)
test_df["text"] = test_df["text"].apply(transformText)
    

# %%
train_df["text"][0:20]

# %%
test_df["text"][0:20]

# %% [markdown]
# ### 1.4 Checking for duplicate rows in the training dataset:

# %%
train_df.duplicated().sum()

# %%
#train_df.drop(axis="rows", labels=train_df.index[train_df.duplicated()], inplace=True) #Removing duplicate rows

# %%
train_df.duplicated().sum()

# %% [markdown]
# ### 1.6 Structuring the data:

# %% [markdown]
# #### 1.6.1 Extracting the labels:

# %%
y_train = train_df.iloc[:, 1:]
label_type = {
    0 : 'harsh',
    1 : 'extremely_harsh',
    2 : 'vulgar',
    3 : 'threatening',
    4 : 'disrespect',
    5 : 'targeted_hate'
}
y_train

# %%
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

my_texts = [] #Consists of texts from both the training and test data
for text in train_df['text']:
    my_texts.append(text.split())
for text in test_df["text"]:
    my_texts.append(text.split())

#vectorSize = int(X_train_tfidf.shape[1]/2)
vectorSize = 10 #word vector size
W2VModel = Word2Vec(sentences=my_texts, vector_size=vectorSize, window=5, min_count=1, workers=4) #Using 'my_texts' to train the Word2Vec model
#vector = W2VModel.wv[word]

# %% [markdown]
# # Creating the datasets

# %%
import torch
from os.path import exists
import os

if not exists("Datasets"):
    os.system("mkdir Datasets")
    

for i in range(6):
    with open("Datasets/"+label_type[i]+"_TrainDataset.txt", "w") as d:
        pass

for j, text in enumerate(train_df['text']): #Creating the training dataset
    seq = [] #stores the sequence of word vectors corresponding to the words in the comment
    for word in text.split():
        seq.append(list(W2VModel.wv[word]))
    for i in range(6):
        with open("Datasets/"+label_type[i]+"_TrainDataset.txt", "a") as d:
            d.write(str(y_train[label_type[i]][j])+":"+str(seq)+"\n")
            

# %%
with open("Datasets/TestDataset.txt", "w") as d:
    pass

for j, text in enumerate(test_df['text']): #Creating the test dataset
    seq = [] #stores the sequence of word vectors corresponding to the words in the comment
    for word in text.split():
        seq.append(list(W2VModel.wv[word]))
    with open("Datasets/TestDataset.txt", "a") as d:
        d.write("2:"+str(seq)+"\n")


