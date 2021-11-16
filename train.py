import random
import json
import numpy as np
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

lem = WordNetLemmatizer()

## This script trains the NN from the intents.json file if nor all_data.pkl file is found.
## Delete all files in "Pickle" to retrain model from updated intents.json

try:
    with open('/pickle/all_data.pkl', 'rb') as file:
        words, classes, training, labels = pickle.load(file)
except:
    intents = json.loads(open('intents.json').read())
    intents = intents['intents']
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lem.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    # pickle.dump(words, open('words.pkl', 'wb'))
    # pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    print(words)
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        # print(doc[0])
        word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]
        # print(word_patterns)
        for word in words:
            if word in word_patterns:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training)
    with open('pickle/all_data.pkl', 'wb') as file:
        pickle.dump((words, classes, training[:, 0], training[:, 1]), file)
# print(training)

# train / test
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))
## model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# compile
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# train
try:
    # don't retrain if model exists
    model = load_model('pickle/chatbotmodel.h5')
    print('model loaded from h5...')
except:
    hist = model.fit(train_x, train_y, epochs=2000, batch_size=5, verbose=1)
    model.save('pickle/chatbotmodel.h5', hist)
    print('model trained from json...')
