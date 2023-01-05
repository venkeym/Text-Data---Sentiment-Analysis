import streamlit as st
import numpy as np
from pickle import load
import re
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

vocab = load(open('models/bow_vocab.pkl','rb'))
lr_model = load(
    open('models/lr_bow_model.pkl', 'rb'))


def preprocess(raw_text, flag):
    # Removing special characters and digits
    sentence = re.sub("[^a-zA-Z]", " ", raw_text)

    # change sentence to lower case
    sentence = sentence.lower()

    # tokenize into words
    tokens = sentence.split()

    # remove stop words
    clean_tokens = [t for t in tokens if not t in stopwords.words("english")]

    # Stemming/Lemmatization
    if (flag == 'stem'):
        clean_tokens = [stemmer.stem(word) for word in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]

    return ([" ".join(clean_tokens)])

text = st.text_input("Review", placeholder="Enter your review :")

btn_click = st.button("Predict")

if btn_click == True:
    if text:

        query_point = preprocess(text, 'lemma')
        query_point_transformed = vocab.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("Enter the values properly.")