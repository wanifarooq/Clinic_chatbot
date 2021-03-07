import json
import numpy as np
from tensorflow import keras
import time
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import colorama
colorama.init()
import speech_recognition as speech
import pyttsx3
import pickle

with open("intents.json") as file:
    data = json.load(file)



def Audio():
    record = speech.Recognizer()
    with speech.Microphone() as source:
        print("please start the satement")
        audio_text = record.listen(source)
        print("querry ended")
        try:
            text = str(record.recognize_google(audio_text))
            print('User :', text)
        except Exception as e:
           print("Oops!", e.__class__, "occurred.")
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[0].id)
    time.sleep(2)
    text = chat(text)
    engine.say(text)
    engine.runAndWait()
    if engine._inLoop:
        engine.endLoop()
    return text

def chat(msg):
    patterns = msg.split()
    pattern = [lemmatizer.lemmatize(x) for x in patterns]
    pattern = ' '.join(pattern)
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 20
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([pattern]), truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])
    for i in data['intents']:
        if i['tag'] == tag:
            return  np.random.choice(i['responses'])