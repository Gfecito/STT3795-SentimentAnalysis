default_params = {
    "root_data" : "data",
    "maxlength": 30,
    "ngram": 1,
    "emb_dim": 300,
    "optimizer": "Adam",
    "lr": 0.001,
    "loss": "binary_crossentropy",
    "batch_size": 1024,
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

def plot_history(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Train accuracy')
    plt.plot(x, val_acc, 'r', label='Val accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Train loss')
    plt.plot(x, val_loss, 'r', label='Val loss')
    plt.title('Training and validation loss')
    plt.legend()


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
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report
    plt.style.use('ggplot')

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

    if model_type == "RNN":
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

        #Model 1
        emb_dim = emb_dim

        inputs = Input(shape=(maxlength, ))

        model1 = Sequential()
        model1.add(Embedding(vocab_size, emb_dim, input_length=maxlength))
        model1.add(SpatialDropout1D(0.2))
        model1.add(Conv1D(32, 5, activation='relu'))
        model1.add(Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4)))
        model1.add(Dense(128, input_dim=input_dim, activation='relu'))
        model1.add(Dropout(0.5))
        model1.add(Dense(64, activation='relu'))
        model1.add(Dropout(0.5))
        model1.add(Dense(2, activation='softmax'))

        opt = optimizer_function(optimizer_string, learning_rate)
        model1.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        #Train
        history = model1.fit(X_train, y_train, 
                        batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test, y_test),
                    )
        plot_history(history=history)
        score = model1.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        preds1 = model1.predict(X_test)
        classes1 = np.argmax(preds1, axis=1)
        print(classification_report(y_test, classes1))

    elif model_type == "LogisticRegression":
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train).astype('float16')
        X_test = vectorizer.transform(X_test).astype('float16')

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        model = LogisticRegression()
        model.fit(X_train, y_train)
        model_score = model.score(X_test, y_test)
        print('Logistic Regression Accuracy:', model_score)

        preds2 = model2.predict_proba(X_test)
        classes2 = np.argmax(preds2, axis=1)
        print(classification_report(y_test, classes2))

    elif model_type == "MLP":
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(X_train).astype('float16')
        X_test = vectorizer.transform(X_test).astype('float16')

        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)
        y_test = encoder.transform(y_test)

        input_dim = X_train.shape[1]  # Number of features

        model3 = Sequential()
        model3.add(Dense(64, input_dim=input_dim, activation='relu'))
        model3.add(Dense(32, activation='relu'))
        model3.add(Dense(2, activation='softmax'))
        opt = optimizer_function(optimizer_string, learning_rate)
        model3.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        history = model3.fit(X_train, y_train, 
                    batch_size=batch_size, epochs=2, 
                   )
        plot_history(history=history)
        score = model3.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0]) 
        print('Test accuracy:', score[1])
        preds3 = model3.predict(X_test)
        classes3 = np.argmax(preds3, axis=1)
        print(classification_report(y_test, classes3))
    
    elif model_type == "Ensemble":
        vectorizer = TfidfVectorizer()
        X_train_vect = vectorizer.fit_transform(X_train).astype('float16')
        X_test_vect = vectorizer.transform(X_test).astype('float16')

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
        y_train_vect = encoder.transform(y_train)
        y_test_vect = encoder.transform(y_test)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #Model 1
        emb_dim = emb_dim

        inputs = Input(shape=(maxlength, ))

        model1 = Sequential()
        model1.add(Embedding(vocab_size, emb_dim, input_length=maxlength))
        model1.add(SpatialDropout1D(0.2))
        model1.add(Conv1D(32, 5, activation='relu'))
        model1.add(Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4)))
        model1.add(Dense(128, input_dim=input_dim, activation='relu'))
        model1.add(Dropout(0.5))
        model1.add(Dense(64, activation='relu'))
        model1.add(Dropout(0.5))
        model1.add(Dense(2, activation='softmax'))

        opt = optimizer_function(optimizer_string, learning_rate)
        model1.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        #Train
        history = model1.fit(X_train, y_train, 
                        batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test, y_test),
                    )
        print("Model1")
        plot_history(history=history)

        #Model 2
        model2 = LogisticRegression()
        model2.fit(X_train_vect, y_train_vect)

        #Model3
        input_dim = X_train_vect.shape[1]  # Number of features

        model3 = Sequential()
        model3.add(Dense(64, input_dim=input_dim, activation='relu'))
        model3.add(Dense(32, activation='relu'))
        model3.add(Dense(2, activation='softmax'))
        opt = optimizer_function(optimizer_string, learning_rate)
        model3.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
        history3 = model3.fit(X_train_vect, y_train_vect, 
                    batch_size=1024, epochs=2, 
                   )
        print("Model3")
        plot_history(history=history3)

        preds1 = model1.predict(X_test)
        classes1 = np.argmax(preds1, axis=1)
        preds2 = model2.predict_proba(X_test_vect)
        classes2 = np.argmax(preds2, axis=1)
        preds3 = model3.predict(X_test_vect)
        classes3 = np.argmax(preds3, axis=1)

        #Majority vote
        final_classes = []
        for i in range(len(classes1)):
            if classes1[i] == classes2[i] and classes2[i] == classes3[i]:
                final_classes.append(classes1[i])
            elif classes1[i] == classes2[i] and classes2[i] != classes3[i]:
                final_classes.append(classes1[i])
            elif classes1[i] == classes3[i] and classes1[i] != classes2[i]:
                final_classes.append(classes1[i])
            elif classes2[i] == classes3[i] and classes2[i] != classes1[i]:
                final_classes.append(classes2[i])
            else:
                final_classes.append(0)

        final_classes = np.array(final_classes)
        print(classification_report(y_test_vect, final_classes))

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
    print("Done")

    # wandb.finish()