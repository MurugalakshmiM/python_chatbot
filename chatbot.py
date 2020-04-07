# things we need for NLP
# Do this in a separate python interpreter session, since you only have to do it once
import nltk

# Do this in your ipython notebook or analysis script
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

botName =raw_input("enter Bot Name : ")
# import our chat-bot intents file
import json
with open(botName+'.json') as json_data:
    intents = json.load(json_data)
with open('entity.json') as entity_data:
    entity = json.load(entity_data)

words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = word_tokenize(pattern)
        entity_list = []
        # add to our words list
        words.extend(w)
        for entityWord in w:
            for en in entity["entity"]:
                if en["tag"] == entityWord:
                    entity_list = en["value"]

        w.extend(entity_list)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
for en in entity['entity']:
    for value in en["value"]: 
        words.append(value)
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

# print (len(documents), "documents",documents)
# print (len(classes), "classes", classes)
# print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    entityList = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # for p_words in doc[0]:
    #     for en in entity['entity']:
    #         if p_words == en["tag"]:
    #             for word in en["value"]:
    #                 pattern_words.append(word)
    #                 # words.append(word)
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # print(pattern_words)
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])




# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    entityList = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    for i,p_words in enumerate(doc[0]):
        for en in entity['entity']:
            if p_words == en["tag"]:
                for word in en["value"]:
                    pattern_words[i] = word
                    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
                    bag = []
                    for w in words:
                        bag.append(1) if w in pattern_words else bag.append(0)
                    output_row = list(output_empty)
                    output_row[classes.index(doc[1])] = 1
                    # print(training)
                    print([bag, output_row], "bag")
                    training.append([bag, output_row])
                    print(training)
                    # words.append(word)
    # # stem each word
    # pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # # print(pattern_words)
    # # create our bag of words array
    # for w in words:
    #     bag.append(1) if w in pattern_words else bag.append(0)

    # # output is a '0' for each tag and '1' for current tag
    # output_row = list(output_empty)
    # output_row[classes.index(doc[1])] = 1

    # training.append([bag, output_row])


# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='restaurant_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save(botName+'.tflearn')
# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( botName+"_training_data", "wb" ) )
print(tflearn.variables.get_all_trainable_variable ())