from flask import Flask, render_template, request

#TODO prevent repeats https://i.imgur.com/OmvDmgk.png

# import sqlite3 as sql
## keras chat brain

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import time

## end keras chat brain

## vars
now = time.time()# float

filename = str(now)+"_chatlog.txt" #create chatlog

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
    
## end vars

class Storage:
    old_answers=[] #storage for answers
    
    @classmethod
    def save_storage(cls):
        with open ("storage.txt", "w") as myfile:
            for answer in Storage.old_answers:
                
                myfile.write(answer+"\n")

    @classmethod
    def load_storage(cls):
        Storage.old_answers=[]
        with open ('storage.txt', 'r') as myfile:
            lines = myfile.readlines()
            for line in lines:
                Storage.old_answers.append(line.strip())
        print (Storage.old_answers)


app = Flask(__name__, template_folder = 'templates')

def bot_response(userText):

    '''fake brain'''
    print ("your q was: " + userText)
    return "your q was: " + userText
   
## new funcs
def clean_up_sentence(sentence):
    """tokenizes the sentences"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    '''read in the intents file'''
    #pseudo code
    #assume old answers are inside
    # old_answers = ['response1','response2']

    #load old answers into storage
    Storage.load_storage()
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    old_answers = Storage.old_answers  # [:-len(list_of_intents)]
    possible_responses = [i['responses'] for i in list_of_intents if i['tag']== tag ][0]
    history = Storage.old_answers[-len('possible_responses'):]
    print("** possible answers and history old answers",possible_responses,history)
    unused_answers = [answer for answer in possible_responses if answer not in history ] # list comprehension
    print(unused_answers, " unused answers")
    unused_two = history[-(len(possible_responses)-1):]
    print(unused_two,'last five answers')
    try:
        result = random.choice([answer for answer in possible_responses if answer not in unused_two ])
    except IndexError:
        print("I'm out of options, I will choose random.")
        result = random.choice(possible_responses)

    Storage.old_answers.append(result) 
    Storage.old_answers= Storage.old_answers[-20:] 
    Storage.save_storage()

    # for i in list_of_intents:
    #     if(i['tag']== tag):
    #         for attempt in range(20):
    #             result = random.choice(i['responses']) 
    #             print("this is results and lookup",result,Storage.old_answers[-len('possible_responses'):])
    #             if result in Storage.old_answers[:-len('possible_responses')]:
    #                continue  
    #             break
    #         # result = random.choice(i['responses'])
    #         print('i found the answer after so many loops',attempt)
    #         print("this is results and lookup",result,Storage.old_answers[-len('possible_responses'):])
    #         Storage.old_answers.append(result) 
    #         Storage.old_answers= Storage.old_answers[-20:] 
    #         Storage.save_storage()
    #         print("*****************")
    #         print ("old answers: ",Storage.old_answers)
    #         break
    return result,tag

def chatbot_response(msg):
    '''this func is important'''
    ints = predict_class(msg, model)
    res,tag = getResponse(ints, intents)
    return res,tag


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
        #append to log file
        with open(filename,'a') as myfile:
            myfile.write("user: "+ msg + "\n")
            myfile.write("bot: "+ res + "\n")

## end new funcs

@app.route('/')
def home():
    return render_template('home.html')

#
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route("/get")
def get_bot_response():    
    print ("get is called")
    userText = request.args.get('msg')    
    # return str(bot.get_response(userText)) 
    # return bot_response(userText)
    res,tag = chatbot_response(userText)
    with open( "logfile.csv", "a" ) as logfile:
        logfile.write(str(now)+","+userText+","+res+","+tag+","+"\n")
        

    return res + '<p style="font-size:8pt;">tag: ' + tag + '</p>'
    



if __name__ == '__main__':
    app.run(debug=True)