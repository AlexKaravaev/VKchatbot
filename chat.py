import numpy as np
import pickle
import tensorflow as tf
#from flask import Flask, jsonify, render_template, request
import model
import codecs

import telebot

# Load in data structures
def read_data(filename):
    with codecs.open(filename, 'r','utf-8') as myfile:
        data=myfile.read().split()
        return data
words = read_data("questions.txt")
words.extend(read_data("responses.txt"))
#print(len(words))
#print(words[100:11000])
wordList = words
wordList.append('<pad>')
wordList.append('<EOS>')

# Load in hyperparamters
vocabSize = len(wordList)
batchSize = 24
maxEncoderLength = 15
maxDecoderLength = 15
lstmUnits = 112
numLayersLSTM = 3

# Create placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)
#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
                                                            vocabSize, vocabSize, lstmUnits, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

# Start session and get graph
sess = tf.Session()
#y, variables = model.getModel(encoderInputs, decoderLabels, decoderInputs, feedPrevious)

# Load in pretrained model
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))
zeroVector = np.zeros((1), dtype='int32')


def pred(inputString):
    inputVector = model.getTestInput(inputString, wordList, maxEncoderLength)
    feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    ids = (sess.run(decoderPrediction, feed_dict=feedDict))
    return model.idsToSentence(ids, wordList)

# webapp
#app = Flask(__name__)

bot = telebot.TeleBot(token)
@bot.message_handler(commands=['start'])
def start(message):
    sent = bot.send_message(message.chat.id, 'Привет! Я обученная переписка с вк нейронная сеть! Поговори со мной')
    bot.register_next_step_handler(sent,send_msg)

# @bot.message_handler(func=lambda m:True)
# def prediction(message):
#     bot.register_next_step_handler(message,send_msg)
#     #print("Бот: " + response)

@bot.message_handler(content_types=["text"])
def send_msg(message):
    bot.send_message(
        message.chat.id,
        pred(message.text))
def main():
    bot.polling(none_stop=True)
    


if __name__ == '__main__':
    main()
    