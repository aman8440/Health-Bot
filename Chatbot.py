import datetime
from plyer import notification
from ctypes import cdll
from http.client import responses
from pyexpat import model
from statistics import mode
from tkinter import W
from turtle import shape
from typing import List
from unittest import result
from urllib import response
from warnings import catch_warnings
from flask import Flask, render_template, redirect, url_for, request
import os
import urllib.request
import nltk
from tflearn import DNN 
#from keras import optimizer_v1
from keras.optimizers import Adam
#from keras.optimizer_experimental import Adam
#from keras import losses 
#from keras import optimizers 
#from keras import metrics 
#from keras.models import compile

from nltk.stem.lancaster import LancasterStemmer
from textblob import Word
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn 
import random
import json
import pickle

"""#For Nortification
def nortify(inp):
    if _name=="main_":
    
		    notification.notify(
			    title = "HEADING HERE",
			    message=" DESCRIPTION HERE" ,
		
			    # displaying time
			    timeout=2
)# waiting time
time.sleep(7)"""

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = np .array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#model.DNN = sequential{}
#model = tflearn.DNN(net)

from tensorflow.python.framework import ops
ops.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[1])])
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, 512)
#net = tflearn.lstm(net,3)
#net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, 512)
net = tflearn.fully_connected(net, len(output[1]), activation="Softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
try:
   model.load('model.tflearn')
except:
    
    # net = tflearn.input_data(shape=[None, len(training[1])])
    # net = tflearn.fully_connected(net, 512)
    # net = tflearn.fully_connected(net, 512)
    # #net = tflearn.lstm(net,3)
    # #net = tflearn.dropout(net, 0.5)
    # net = tflearn.fully_connected(net, 512)
    # net = tflearn.fully_connected(net, len(output[1]), activation="Softmax")
    # net = tflearn.regression(net)
    # model = tflearn.DNN(net)
    # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.fit(training, output, n_epoch=100, batch_size=64, show_metric=True)
    #optimizers.adam_v2(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adam = tf.optimizers_v1.
    #Adam(0.001)
    
    #tf.losses('categorical_crossentropy')
    #model.compile(optimizer= Adam,losses='categorical_crossentropy',metrics=['accuracy'])
                                        
    model.save('model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

# app = Flask(__name__,static_url_path='/static')   
# @app.route('/success/<name>')
# def success(name):
#    return 'welcome %s' % name
# @app.route('/chat',methods = ['POST', 'GET'])
def chat():
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]


        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print("Ishita: "+random.choice(responses))        
#     if request.method == 'POST':
#         print(request)
#         user = request.form['messgae']
#         return redirect(url_for('success',name = user))
#     else:
#         user = request.args.get('message')
#         return redirect(url_for('success',name = user)) 
           
#         #print(results)
# chat()
# @app.route('/')
# def index():
#     return render_template('index.html',len=len('intents')) 
# if __name__ == '__main__':
#   app.run(use_reloader = True, debug=True)
def main():
# open a connection to a URL using urllib2
   webUrl = urllib.request.urlopen('index.html')
   data = webUrl.read()
   print(data)
 
if __name__ == "__main__":
  main()