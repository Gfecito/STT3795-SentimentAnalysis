default_params = {
    "root_data" : "data",
    "maxlength": 30,
    "ngram": 1,
    "emb_dim": 300,
    "optimizer": "Adam",
    "lr": 0.001,
    "loss": "binary_crossentropy",
    "batch_size": 16,
    "epochs": 10,
    "model": "RNN",
}

def defaults(dictionary, dictionary_defaults):
    for key, value in dictionary_defaults.items():
        if key not in dictionary:
            dictionary[key] = value
        else:
            if isinstance(value, dict) and isinstance(dictionary[key], dict):
                dictionary[key] = defaults(dictionary[key], value)
            elif isinstance(value, dict) or isinstance(dictionary[key], dict):
                raise ValueError("Given dictionaries have incompatible structure")
    return 

def optimizer_function(type, lr):
    import tensorflow as tf
    from tensorflow import keras

    if type == "Adam":
        return keras.optimizers.Adam(lr=lr)
    elif type == "SGD":
        return keras.optimizers.SGD(lr=lr)
    else:
        raise ValueError("Optimizer not implemented or naming error")

def cleaning(a):
    import re
    import string

    a = str(a).lower()
    a = re.sub('\[.*?\]', '', a)
    a = re.sub('[%s]' % re.escape(string.punctuation), '', a)
    a = re.sub('\n', '', a)
    a = re.sub('https?://\S+|www\.\S+', '', a)
    a = re.sub('<.*?>+', '', a)
    a = re.sub('\w*\d\w*', '', a)
    return a

def stemSentence(sentence):
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import PorterStemmer

    porter=PorterStemmer()

    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def stem(x_train):
    for i in range(len(x_train)):
        x_train[i] = stemSentence(x_train[i])

    return x_train

def experiment(params):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Dropout, Layer
    from tensorflow.keras.layers import Embedding, Input, GlobalAveragePooling1D, Dense
    from tensorflow.keras.models import Sequential, Model
    from keras.layers import LSTM, Flatten, Conv1D, MaxPool1D, Bidirectional, Embedding, SpatialDropout1D, Attention, BatchNormalization, GlobalMaxPool1D, GRU, SimpleRNN, ConvLSTM1D
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    root_data = params["root_data"]
    maxlength = params["maxlength"]
    ngram = params["ngram"]
    emb_dim = params["emb_dim"]
    optimizer_string = params["optimizer"]
    learning_rate = params["lr"]
    loss = params["loss"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    model_type = params["model"]

    df = pd.read_csv(root_data, encoding='latin-1', header=None)

    label_sentiment_dict = {0: "Negative", 4: "Positive"}

    def label_map(a):
        return label_sentiment_dict[a]

    df[0] = df[0].apply(label_map)

    df[5] = df[5].apply(cleaning)

    x_train = df[5].to_numpy()
    y_train = df[0].to_numpy()

    x_train = stem(x_train)

    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.02, random_state=22)

    #Tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)
    vocab_size = len(tokenizer.word_index) + 1
    texts_to_int = tokenizer.texts_to_sequences(X_train)
    #Padding
    texts_to_int_pad = keras.preprocessing.sequence.pad_sequences(texts_to_int,
                                                                maxlen=maxlength,)
    texts_to_int_test = tokenizer.texts_to_sequences(X_test)
    texts_to_int_pad_test = keras.preprocessing.sequence.pad_sequences(texts_to_int_test,
                                                                maxlen=maxlength,)
    
    X_train = texts_to_int_pad
    X_test = texts_to_int_pad_test

    #label encoding
    labels = ["Negative", "Positive"]
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)


    if model_type == "RNN":
    #Model 1
        emb_dim = emb_dim

        inputs = Input(shape=(maxlength, ))

        embedding_layer = Embedding(vocab_size, emb_dim, input_length=maxlength)
        x = embedding_layer(inputs)
        x = SpatialDropout1D(0.2)(x)
        x = Conv1D(32, 5, activation='relu')(x)
        # Passed on to the LSTM layer
        x = Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4))(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        # Passed on to activation layer to get final output
        out = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=out)
        opt = optimizer_function(optimizer_string, learning_rate)
        model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        #Train
        history = model.fit(X_train, y_train, 
                        batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test, y_test),
                    )
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])

    elif model_type == "LogisticRegression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        model_score = model.score(X_test, y_test)
        print('Logistic Regression Accuracy:', model_score)
    

if __name__ == "__main__":
    import json
    import argparse
    # import wandb

    # from datetime import datetime

    # now = datetime.now()

    # current_time = now.strftime("%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", "-p", type=str, help="JSON params file")
    parser.add_argument("--direct", "-d", type=str, help="JSON state string")
    
    arguments = parser.parse_args()
    
    if arguments.direct is not None:
        params = json.loads(arguments.direct)
    elif arguments.params is not None:
        with open(arguments.params) as file:
            params = json.load(file)
    else:
        params = {}

    params = defaults(params, default_params)
    # log_name = params["dataset"] + "-" + current_time
    # wandb.init(project="CNN-Magic", name = log_name, entity="qinjerem", config=params)

    experiment(params)

    # wandb.finish()