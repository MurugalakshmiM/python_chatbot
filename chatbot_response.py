botName =raw_input("enter Bot Name : ")
print(botName+'.json')
# restore all of our data structures
import pickle
data = pickle.load( open( botName+"_training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
import tflearn
import nltk

# Do this in your ipython notebook or analysis script
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import random

# import our chat-bot intents file
import json
with open(botName+'.json') as json_data:
    intents = json.load(json_data)

# load our saved model
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir=botName+'_logs')
# load our saved model
model.load('./'+botName+'.tflearn')
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    print(bag,"bag")
    return(np.array(bag))

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    print(words)
    results = model.predict([bow(sentence, words)])[0]
    print(results)
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        print(random.choice(i['responses']))
                        return random.choice(i['responses'])

            results.pop(0)
            if i['tag'] == results[0][0]:
                # set context for this intent if necessary
                if 'context_set' in i:
                    if show_details: print ('context:', i['context_set'])
                    context[userID] = i['context_set']

                # check if this intent is contextual and applies to this user's conversation
                if not 'context_filter' in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                    if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                    print(random.choice(i['responses']))
                    return random.choice(i['responses'])
exitCondition = '' 
while exitCondition != 'exit' : 
    convo =raw_input("input chat: ") 
    response(convo) 
    exitCondition = convo
