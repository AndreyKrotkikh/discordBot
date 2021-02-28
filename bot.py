import discord
import tensorflow
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Embedding, Dense, Flatten, LSTM, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import pickle
import io
import json
import os
import time
import heapq
from past.builtins import xrange
import re
import sys
import random


class neuralNet():

    def __init__(self):
        ## load tokenizer
        with open('tokenizer.json') as f:
            data = json.load(f)
            self.tokenizer = tokenizer_from_json(data)

        with open('tokenizerForNames.json') as f:
            data = json.load(f)
            self.tokenizerForNames = tokenizer_from_json(data)

        ## load tokenizer2
        with open('tokenizer2.json') as f:
            data = json.load(f)
            self.tokenizer2 = tokenizer_from_json(data)

        with open('tokenizer2ForNames.json') as f:
            data = json.load(f)
            self.tokenizer2ForNames = tokenizer_from_json(data)

        print('tokenizers2 on')

        ## model load

        checkpoint_path = "training_1/cp.ckpt"

        self.model = Sequential()
        self.model.add(Embedding(10000, 64, input_length=40))
        self.model.add(Conv1D(32, 8, activation='relu'))
        self.model.add(MaxPooling1D(5))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Flatten())
        self.model.add(Dense(57, activation='softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model.load_weights(checkpoint_path)
        print('model on')

        ## model load

        checkpoint_path = "training_2/cp.ckpt"

        self.model2 = Sequential()
        self.model2.add(Embedding(10000, 64, input_length=50))
        self.model2.add(Conv1D(32, 8, activation='relu'))
        self.model2.add(MaxPooling1D(5))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(LSTM(64, return_sequences=True))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(LSTM(64, return_sequences=True))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(Dense(100, activation='relu'))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(Dense(10, activation='relu'))
        self.model2.add(BatchNormalization())
        self.model2.add(Dropout(0.3))
        self.model2.add(Dense(1, activation='sigmoid'))
        self.model2.compile(optimizer='rmsprop', loss='binary_crossentropy')
        #self.model2.load_weights(checkpoint_path)
        print('model2 on')

        ## model load

        self.maxlen = 60
        with open('chars_list.data', 'rb') as filehandle:
            # read the data as binary data stream
            self.chars = pickle.load(filehandle)
        print('Unique characters:', len(self.chars))

        print(self.chars)

        self.char_indices = dict(((char, self.chars.index(char)) for char in self.chars))

        checkpoint_path = "training_3/cp.ckpt"

        self.model3 = Sequential()
        self.model3.add(LSTM(256, input_shape=(self.maxlen, len(self.chars)), return_sequences=True))
        self.model3.add(LSTM(256, return_sequences=True))
        self.model3.add(LSTM(256))
        self.model3.add(Flatten())
        self.model3.add(Dense(len(self.chars), activation='softmax'))
        self.model3.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model3.load_weights(checkpoint_path)
        print('model3 on')

    def classify(self, string):
        sequence = self.tokenizer.texts_to_sequences([string])
        preprocessedsequence = preprocessing.sequence.pad_sequences(sequence, maxlen=40)
        predictions = self.model.predict(preprocessedsequence)
        position = numpy.argmax(predictions) + 1
        #print(predictions.tolist()[1])
        #print([predictions.tolist()[0].index(x) for x in sorted(predictions.tolist()[0], reverse=True)[:3]])
        print(sum(predictions[0]))
        names = []
        nameprobabilities = []
        for tokenPosition in [predictions.tolist()[0].index(x) for x in sorted(predictions.tolist()[0], reverse=True)[:3]]:
            names.append(self.tokenizerForNames.sequences_to_texts([[tokenPosition + 1]]))
            nameprobabilities.append(predictions[0][tokenPosition])
        #print(nameprobabilities)
        return [names, nameprobabilities]

    def react(self, msgs):
        sequence = self.tokenizer2.texts_to_sequences(msgs)
        preprocessed = preprocessing.sequence.pad_sequences(sequence, maxlen=10)
        joined = numpy.array(preprocessed).flatten()
        result = self.model2.predict(numpy.array([joined]))
        return result

    def sample(self, preds, temperature):
        preds = numpy.asarray(preds).astype('float64')
        preds = numpy.log(preds) / temperature
        exp_preds = numpy.exp(preds)
        preds = exp_preds / numpy.sum(exp_preds)
        probas = numpy.random.multinomial(1, preds, 1)
        return numpy.argmax(probas)

    def continue_msg(self, msg, temperature, msglength):
        generated_text = msg[:60]
        result = ''
        if len(generated_text) >= 60:
            print(len(generated_text))
            sys.stdout.write(generated_text)
            for i in range(msglength):
                sampled = numpy.zeros((1, self.maxlen, len(self.chars)))
                for t, char in enumerate(generated_text):


                    #print('+ ', str(char))
                    sampled[0, t, self.char_indices[char]] = 1.
                    #print('+ ', sampled)
                preds = self.model3.predict(sampled, verbose=0)[0]
                #print('+ ', preds)
                next_index = self.sample(preds, temperature)
                next_char = self.chars[next_index]
                #print(next_index)
                generated_text += next_char
                result += next_char
                generated_text = generated_text[1:]
                sys.stdout.write(next_char)
            print(result)
            return result
        else:
            return ('Системное сообщение: Мало текста для ответа')


class MyClient(discord.Client):

    async def on_ready(self):
        print('Logged on as {0}!'.format(self.user))
        for guild in self.guilds:
            print('Logged on {0}'.format(guild.name))

    async def on_message(self, message):
        # print(message.author.name)
        # print(message.author.id)
        if False:#not(message.content.startswith('++')) and str(message.channel) == 'урал-вагон-завод':
            channel = message.channel
            train_data = open("train.txt", "a")
            try:
                train_data.write(str(message.author.name) + "\\" + message.content + " \n")
            except:
                print("Caught exception")
            train_data.close()

            if str(message.channel) == 'урал-вагон-завод':
                print(neural.classify(str(message.content)))
            # await channel.send(content=message.content)

        if False: #message.content.startswith('++classify ') and message.author.id == 366921098077667338:
            channel = message.channel
            print(message.content[11:])
            if len(message.content[11:].split(' ')) <= 4:
                await channel.send(content='Малое количество слов в последовательности')
            else:
                await channel.send(content=neural.classify(str(message.content[11:])))

        if message.content.startswith('++загрузитьисторию') and message.author.id == 366921098077667338:
            channel = message.channel
            await channel.send(content='Начинается загрузка данных')

            startTime = time.time()
            messages = await channel.history(limit=None).flatten()

            train_data = open("train3.txt", "a", errors="surrogateescape")
            for msg in reversed(messages):
                try:

                    print(str(msg.author.name))
                    if True:#str(msg.author.name) == 'Giallar':
                        train_data.write(re.sub(r'<[^()]*>', '', msg.clean_content) + " ")
                except:
                    print("Caught exception")
            train_data.close()

            await channel.send(content='Данные загружены ' + str((time.time() - startTime)/60))

        mentioncheck = False
        for mention in message.mentions:
            if mention.name == 'TradeBot' and message.author.id == 366921098077667338:
                mentioncheck = True
                break

        if mentioncheck:
            channel = message.channel
            messages = await channel.history(limit=50).flatten()
            msglist = numpy.array([])
            for msg in messages:
                msglist = numpy.append(msglist, msg.content)
            print('\n Тест ответа \n')
            joined_msg = ' '.join(msglist)
            test = map(str.strip, str.split(''.join(filter(lambda x: x.isalnum or x == ' ', joined_msg)), '\n'))
            test = ' '.join([str(x) for x in test])
            joined_msg = re.sub(r'<[^()]*>', '', test)
            #print(joined_msg.lower())
            messagelen = random.randint(50, 100)
            print(messagelen)
            print(len(joined_msg.lower()))
            if len(joined_msg.lower()) >= 60:
                await channel.send(content=neural.continue_msg(joined_msg.lower(), 0.2, messagelen))

        if str(message.channel) == 'урал-вагон-завод' or str(message.channel) == 'бар-у-завода':
            channel = message.channel
            messages = await channel.history(limit=5).flatten()
            msglist = numpy.array([])
            for msg in messages:
                msglist = numpy.append(msglist, msg.content)
            #neuralsignal = neural.react(msglist)[0][0]
            #print('Нейронный сигнал: ' + str(neuralsignal) + '\n')
            if True:#neuralsignal > 0.10:
                messages = await channel.history(limit=50).flatten()
                msglist = numpy.array([])
                for msg in messages:
                    msglist = numpy.append(msglist, msg.content)
                check = 0
                for msg in messages:
                    if msg.author.name == 'TradeBot':
                     check += 1
                print(str(check))
                chance_generator = random.random()
                print(chance_generator)
                if chance_generator > 0.9:
                    print('\n Тест ответа \n')
                    joined_msg = ' '.join(msglist)
                    test = map(str.strip, str.split(''.join(filter(lambda x: x.isalnum or x == ' ', joined_msg)), '\n'))
                    test = ' '.join([str(x) for x in test])
                    joined_msg = re.sub(r'<[^()]*>', '', test)
                    #print(joined_msg.lower())
                    messagelen = random.randint(50, 100)
                    print(messagelen)
                    if len(joined_msg.lower()) >= 60:
                        await channel.send(content=neural.continue_msg(joined_msg.lower(), 0.2, messagelen))
                    #print(neural.continue_msg(joined_msg.lower(), 1.))

        if message.content.startswith('++generate') and message.author.id == 366921098077667338:
            channel = message.channel
            messages = await channel.history(limit=50).flatten()
            msglist = numpy.array([])
            for msg in messages:
                msglist = numpy.append(msglist, msg.content)
            print('\n Тест ответа \n')
            joined_msg = ' '.join(msglist)
            test = map(str.strip, str.split(''.join(filter(lambda x: x.isalnum or x == ' ', joined_msg)), '\n'))
            test = ' '.join([str(x) for x in test])
            joined_msg = re.sub(r'<[^()]*>', '', test)
            #print(joined_msg.lower())
            messagelen = random.randint(100, 150)
            print(messagelen, ' ', len(joined_msg.lower() ))
            if len(joined_msg.lower()) >= 60:
                await channel.send(content=neural.continue_msg(joined_msg.lower(), 0.1, messagelen))

        #print('Message from {0.author} on {0.guild} on {0.channel}: {0.content}'.format(message))

neural = neuralNet()
client = MyClient()
client.run('NTg5NDQxOTgzNTA5MTY4MTI4.XQTuoA.n8SbAoYYnb4YX0OmKfQcwJyZG2U')
