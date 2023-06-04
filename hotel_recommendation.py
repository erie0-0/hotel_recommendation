

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import os
import string

"""Load Data"""

Df = pd.read_csv("/content/Hotel_data.csv")
Df.head()

"""DATA Cleaning"""

Df = Df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation)) if isinstance(text, str) else text
Df = Df.applymap(remove_punctuation)

# Count the no. of rows and columns
Df.shape

#Count of the empty values in each column
Df.isna().sum()

#Drop the column with all missing values

Df=Df.dropna()

Df = Df.drop([ 'From_Date', 'To_Date','Location','Offer','Discount'],axis = 1)

Df

Df.shape

Df['Special'].unique()

Df.dtypes

l=list()
for i in Df['Price']:
  i = float(i.replace(',', ''))
  l.append(i)
Df['Price']=l

Df.dtypes

Df["Rating"].describe()

Df['Review'].value_counts()

"""DATA VISUALISATION

"""

sns.countplot(x = Df['Review'], label = 'Review Count')

#create a pair plot
sns.pairplot(Df.iloc[:,1:], hue ='Review')

#correlation
Df.iloc[:,1:].corr()*100

#visualize correlation
plt.figure(figsize= (10,10))
sns.heatmap(Df.iloc[:,1:].corr(), annot = True, fmt = '.0%')

!pip install tensorflow 
!pip install tensorflow-text

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
import tensorflow_text as text

preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",trainable=True)

def get_bert_embeddings(text, preprocessor, encoder):

  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  encoder_inputs = preprocessor(text_input)
  outputs = encoder(encoder_inputs)
  embedding_model = tf.keras.Model(text_input, outputs['pooled_output'])
  sentences = tf.constant([text])
  return embedding_model(sentences)

Df['encodings'] = Df['Special'].apply(lambda x: get_bert_embeddings(x, preprocessor, encoder))

import re
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text():
  text = input("Enter the facilities you would like in your hotel : ")
  text = text.lower()
  text = re.sub('[^A-Za-z0-9]+', ' ', text)
  return text
  
query_text = preprocess_text()
query_encoding = get_bert_embeddings(query_text, preprocessor, encoder)

def recommend_hotel(city, price, user_preference):
   
    df_filtered = Df[(Df['City'] == city) & (Df['Price'] <= price)]
    if len(df_filtered) > 0:
        
        df_filtered['similarity_score'] = df_filtered['encodings'].apply(lambda x: metrics.pairwise.cosine_similarity(x, tf.reshape(user_preference, [1,-1]))[0][0])
        df_results = df_filtered.sort_values(by=['Rating', 'similarity_score'], ascending=False)
        print(df_results.head(5))
    else:
        print("No hotels found in the given city and price range")
  
city = input("Enter the city: ").lower()
price = int(input("Enter the price: "))


print("Top 5 hotels:")
recommend_hotel(city, price, query_encoding)