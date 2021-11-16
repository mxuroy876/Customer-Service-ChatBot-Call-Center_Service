# MAUROY FOSTER
#APL FINAL PAPER
# 1700452

import json
import random
import pickle
from nltk import text
import numpy as np
from PIL import ImageTk
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
lem = WordNetLemmatizer()
import spacy
import pyttsx3
import tkinter as tk
import speech_recognition as sr


nlp = spacy.load('en_core_web_sm')
engine = pyttsx3.init()

intents = json.loads(open('intents.json').read())
model = load_model('pickle/chatbotmodel.h5')
with open('pickle/all_data.pkl', 'rb') as file:
    words, classes, training, labels = pickle.load(file)

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lem.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    # print(res)
    error_thresh = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_thresh]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    # print(return_list)
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    prob = float(intents_list[0]['probability'])
    list_of_intents = intents_json['intents']
    error_thresh = 0.60

    if prob > error_thresh:
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
    # if the error_thresh is not met respond IDK
    else:
        # the first item in intents should be responses to unknown inputs
        result = random.choice(intents_json['intents'][0]['responses'])
    return result

bot_name = 'UTECH ChatBot'
from tkinter import *
import time


BG_GRAY = "#e8ebea"
BG_COLOR = "#1670E9"
TEXT_COLOR = "#EAECEE"
TEXT_COLORS ="#051326"
FONT = "Verdana 11"
FONT_BOLD = "Verdana 12"


class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("UTECH CHATBOT")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=650, height=500, bg=BG_COLOR)
        
        head_label = Label(self.window,bg="#0a0a0a", fg=TEXT_COLOR,
                           text="UTECH Customer Service ChatBot Call Center Service", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

     
        # text widget
        self.text_widget = Text(self.window, width=60, height=2,bg=TEXT_COLORS, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)                 
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.984)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=85)
        bottom_label.place(relwidth=1, rely=0.825)
        #img = tk.PhotoImage(file="logo.png")

        # message entry box
        self.msg_entry = Entry(bottom_label, bg="#0a0a0a", fg=TEXT_COLOR, justify="center", font=FONT)
        self.msg_entry.place(relwidth=0.90, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed, self.talk)
        #self.msg_entry.bind("<Return>", self.speech_to_text)

        #ask question button
        ask_question_button = Button(bottom_label, text="Ask Question", font=FONT, bg=BG_GRAY,
                            command=lambda: self._on_enter_pressed(None))
        ask_question_button.place(relx=0.78, rely=0.008, relheight=0.06, relwidth=0.22)

       #read button
        read_button = Button(bottom_label, text="Read Text", font=FONT, justify="center", bg=BG_GRAY,
                             command=lambda: self.talk(None))
        read_button.place(rely=0.008, relheight=0.04, relwidth=0.22)

        #speech to text button
        speech_button = Button(bottom_label, text="Speech to Text", font=FONT, bg=BG_GRAY,
                             command=lambda: self.speech_to_text(None))
        speech_button.place(rely=0.035, relheight=0.04, relwidth=0.22)

    def talk(self, event):
        engine = pyttsx3.init()  # initialize speech engine
        engine.say(self.msg_entry.get())  # allows text to speech
        engine.runAndWait()
        self.msg_entry.delete(0, END)

    #For speech recognistion
    def speech_to_text(self, event):
        r=sr.Recognizer()
        #self.msg_entry.insert(END, "Please Talk")
        with sr.Microphone() as source:
    # read the audio data from the default microphone
            audio_data = r.record(source, duration=5)
        #self.msg_entry.insert(END, "Recognizing...\n\n")
    # convert speech to text
        text = r.recognize_google(audio_data)
        #text = r.recognize_google(audio_data, language="es-ES")
        self.msg_entry.insert(END, text)


    def _on_enter_pressed(self, event):  # event will take the argument entry
        msg = self.msg_entry.get()  # get input text as string
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, END)  # delete msg from txt_entry window
        msg1 = f"{sender}: {msg}\n\n"  # The users message
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        ints = predict_class(msg)
        res = get_response(ints, intents)
        time.sleep(1)
        msg2 = f"{bot_name}: {res}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
