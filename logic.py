import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D,Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def train():
    with open('intents.json') as file:
        data = json.load(file)

    training_sentences = []
    training_labels = []
    labels = []
    responses = []

    for intent in data['intents']:
        for patterns in intent['patterns']:
            patterns = patterns.split()
            pattern = [lemmatizer.lemmatize(x) for x in patterns]
            pattern = ' '.join(pattern)
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"
    tokenizer = Tokenizer( num_words=vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=' ',
                           char_level=False,oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D(data_format='channels_first'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
    epochs = 10000
    model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("chat_model")

    import pickle

    # to save the fitted tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # to save the fitted label encoder
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)


    return "successfully trained"

# train()













