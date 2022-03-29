# coding=utf8
import json
import os
import requests
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from langdetect import detect
from argostranslate import package as a_package, translate as a_translate

from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, tokenizer_from_json

import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')


class Neuronet:
    def __init__(self, path_to_files):
        self.num_words = 25000
        self.max_text_len = 500
        self.model_cnn = Sequential()
        self.model_cnn.add(Embedding(self.num_words, 1000, input_length=self.max_text_len))
        self.model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
        self.model_cnn.add(GlobalMaxPooling1D())
        self.model_cnn.add(Dense(128, activation='relu'))
        self.model_cnn.add(Dense(40, activation='softmax'))

        RequiredFiles.check_files(path_to_files)

        self.load_translator(path_to_files)
        self.load_weights(path_to_files)
        self.load_tokenizer(path_to_files)
        self.read_topics(path_to_files)

    def load_translator(self, path):
        path_to_model = os.path.join(path, 'translate-en_ru-1_1.argosmodel')
        a_package.install_from_path(path_to_model)
        installed_languages = a_translate.get_installed_languages()
        self.translation_en_ru = installed_languages[0].get_translation(installed_languages[1])

    def load_weights(self, path):
        weights_path = os.path.join(path, "best_conv_model.h5")
        self.model_cnn.load_weights(weights_path)

    def load_tokenizer(self, path):
        tokenizer_path = os.path.join(path, "tokenizer.json")
        self.tokenizer = Tokenizer()
        with open(tokenizer_path) as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)

    def read_topics(self, path):
        topics_path = os.path.join(path, "Список тем (utf-8).csv")
        topics = pd.read_csv(topics_path,
                             header=None,
                             sep=";",
                             names=['Class', 'Name'])
        df = pd.DataFrame({"Class_ordered": [i for i in range(40)]})
        self.topics_ordered = df.join(topics)

    def analise(self, text):
        lang = detect(text)
        if lang == "en":
            text = self.translation_en_ru.translate(text)

        sequence = self.tokenizer.texts_to_sequences([text])
        data = pad_sequences(sequence, maxlen=self.max_text_len)

        result = self.model_cnn.predict(data)
        num = np.where(result == np.max(result))[1][0]

        topic = self.topics_ordered[self.topics_ordered['Class_ordered'] == num]['Name'].values[0]

        return lang, topic


class RequiredFiles:
    @staticmethod
    def check_files(path):
        if not os.path.exists(path):
            os.mkdir(path)

        weights_path = os.path.join(path, "best_conv_model.h5")
        tokenizer_path = os.path.join(path, "tokenizer.json")
        topics_path = os.path.join(path, "Список тем (utf-8).csv")
        translate_model_path = os.path.join(path, "translate-en_ru-1_1.argosmodel")

        if not os.path.exists(weights_path):
            link = "https://drive.google.com/uc?id=1oJWoNXqrt4ZBdpZfY96j6r6BJ4gDSraK&export=download&confirm=t"
            print("Weights < file is missing. Attempting to download...")
            RequiredFiles.get_file(link, weights_path)

        if not os.path.exists(tokenizer_path):
            link = "https://drive.google.com/uc?id=1yd3nAT8-9_ONauGBh7sxoZjk1qGvBbY1&export=download&confirm=t"
            print("Tokenizer < file is missing. Attempting to download...")
            RequiredFiles.get_file(link, tokenizer_path)

        if not os.path.exists(topics_path):
            link = "https://drive.google.com/uc?id=1mAgp96OUHUBLaYkpQv5yuR1eEqBttb1A&export=download&confirm=t"
            print("Topics < file is missing. Attempting to download...")
            RequiredFiles.get_file(link, topics_path)

        if not os.path.exists(translate_model_path):
            link = "https://drive.google.com/uc?id=1IpE_9mTKNHrdziajCBzaWgtthrp5i6nC&export=download&confirm=t"
            print("Translation model < file is missing. Attempting to download...")
            RequiredFiles.get_file(link, translate_model_path)

    @staticmethod
    def get_file(link, path):
        with requests.get(link, stream=True) as r:

            total_length = int(r.headers.get("Content-Length"))

            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:

                with open(path, 'wb') as output:
                    shutil.copyfileobj(raw, output)


path_to_net_files = os.path.join(os.getcwd(), 'net_files')
nnet = Neuronet(path_to_net_files)