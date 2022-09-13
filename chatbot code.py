from pyexpat import model
from statistics import mode
from tkinter import W
from turtle import shape
from typing import List
from unittest import result
from warnings import catch_warnings
import nltk
from nltk.stem.lancaster import LancasterStemmer
from textblob import Word
stemmer = LancasterStemmer()

import numpy as np
import tensorflow
import tflearn
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","rb") as f:
        words,labels,training,output =pickle.load(f)
except: 

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intents in data ["intents"]:
        for pattern in intents["patterns"]:
            wrds =nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intents["tags"])

            if intents["tags"] not in labels:
                labels.append(intents["tags"])

    words =[stemmer.stem (W.lower()) for  w in words if W not in "?"]
    words =sorted(List(set(words)))

    labels =sorted(labels)

    training =[]
    output = []

    out_empty =[0 for _ in range (len (labels))]

    for x, doc in enumerate(docs_x) :
        bag = []

        wrds = [stemmer.stem(W) for w in doc]

        for W in words :
            if W in wrds:
                bag.append(1)
            else:
                bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] =1

        training.append(bag)
        output.append(output_row)

    traning = np.array(training)
    output = np.array(output)

    with open("data.pickle","rb") as f:
        pickle.dump(words,labels,training,output,f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape = [None,len(traning[0])])
net= tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training,output,n_epach =1000, batch_size=8, show_metric= True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words)) ]
    s_words =nltk.word_tokenize(s)
    s_words = [stemmer.stem(Word.lower()) for word in s_words]

    for se in s_words:
        for i ,W in enumerate(words):
            if W == se:
                bag[i] == 1
    return np.array(bag)

def chat():
    print("Start talking with the bot")
    while True:
        inp = input("You")
        if inp.lower() == "quit":
            break

        model.predict([bag_of_words(inp,words)])
        print(result)