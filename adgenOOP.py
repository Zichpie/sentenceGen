from __future__ import print_function


import numpy as np
import gensim
import string
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation, Bidirectional
from keras.models import Sequential
from keras import regularizers
from keras.utils.data_utils import get_file
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
nltk.download()



def max_len(list_of_sentence):
    max_length = 0
    for s in list_of_sentence:
        if len(s) > max_length:
            max_length = len(s)
    return max_length

def padding(list_of_sentence, max_length):
    for sent in list_of_sentence:
        if len(sent) < max_length:
            while len(sent) < max_length:
                sent.append('')
    return list_of_sentence

def tokenize(sentences_lowered):
    list_of_sentence = []
    for sent in sentences_lowered:
        tokens = word_tokenize(sent)
        list_of_sentence.append(tokens)
    return list_of_sentence

def process_doc(docs):
    doc1 = docs.replace('"', '')
    doc2 = doc1.replace('par', '')
    doc3 = doc2.replace('rdblquote', '')
    doc4 = doc3.replace('ldblquote', '')
    doc5 = doc4.rstrip('x\00')
    doc6 = doc5.rstrip('d\sa200\sl276\slmult1\cf0\f1\fs22')
    return doc6




class textDataset ():

    def __init__(self, filename):
        self.filename = filename


    def process_dataset(self):
        with open(self.filename) as file_:
            self.docs = file_.read()

            self.processed_doc = process_doc(self.docs)
            self.sentences = sent_tokenize(self.processed_doc)
            self.sentences_lowered = []

            for sent in self.sentences:
                self.sentence_lower = sent.lower().translate(str.maketrans('','',string.punctuation))
                self.sentences_lowered.append(self.sentence_lower)

        self.sentences_lowered[-1] = 'let the beauty of what you love be what you do'
        self.sentences_lowered[0] = 'think big'
        self.max_length = max_len(tokenize(self.sentences_lowered))
        self.list_of_sentence = padding(tokenize(self.sentences_lowered), self.max_length)

        return self.list_of_sentence




dataset1 = textDataset('adtext.rtf')
dataset1 = dataset1.process_dataset()


class WordEmbedding():

    def __init__(self, dataset, min_count, window, workers):
        self.dataset = dataset
        self.min_count = min_count
        self.window = window
        self.workers = workers

    def embed_word(self):
        self.word_model = gensim.models.Word2Vec(self.dataset,
                                             min_count = self.min_count,
                                             window = self.window,
                                             workers = self.workers)

        self.pretrained_weights = self.word_model.wv.vectors
        self.vocab_size, self.embedding_size = self.pretrained_weights.shape
        self.shape_vec = np.shape(self.word_model.wv.vectors[0])
        return self.word_model, self.vocab_size, self.embedding_size, self.pretrained_weights

    def visualizeW(self, word):
        return self.word_model.wv[word]



wordemb = WordEmbedding(dataset1,1,5,4)
word_model, vocab_size, embedding_size, pretrained_weights = wordemb.embed_word()


def word2idx(word, word_model):
    return word_model.wv.vocab[word].index

def idx2word(idx, word_model):
    return word_model.wv.index2word[idx]


class trainingset():

    def __init__(self, los, maxlen, wordModel):
        self.los = los
        self.maxlen = maxlen
        self.wordModel = wordModel

    def init_training_set(self):
        self.train_x = np.zeros([len(self.los), self.maxlen], dtype = np.int32)
        self.train_y = np.zeros([len(self.los)], dtype = np.int32)

        for i, sent in enumerate(self.los):
            for t,word in enumerate (sent[:-1]):
                self.train_x[i,t] = word2idx(word, self.wordModel)
                self.train_y[i]   = word2idx(sent[-1], self.wordModel)
        print( "shape of training_x is:", self.train_x.shape)
        print("shape of training_y is:", self.train_y.shape)

        return self.train_x, self.train_y



maxlen = max_len(dataset1)
trainset1 = trainingset(dataset1, maxlen, word_model)
train_x, train_y = trainset1.init_training_set()


def model_init(model_type, vocab_size, embedding_size, pretrained_weights):

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim= embedding_size, weights=[pretrained_weights]))

    if model_type == "bidirectional":

        model.add(Bidirectional(LSTM(units = embedding_size, return_sequences=True)))
        model.add(Bidirectional(LSTM(units = embedding_size)))

    elif model_type == "uni":
        model.add(LSTM(units = embedding_size))
    else:
        raise Exception("enter a valid model_type")

    model.add(Dense(units = vocab_size))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model



def generate_next(text, word_model, model, num_generated, temperature):
    word_idxs = [word2idx(text, word_model)]
    for i in range(num_generated):
        prediction = model.predict(x=np.array(word_idxs))
        idx = sample(prediction[-1], temperature)
        word_idxs.append(idx)
    return ' '.join(idx2word(idx, word_model) for idx in word_idxs)

# sampling what the next word is going to be
def sample(preds, temperature):
    if temperature <= 0:
        return np.argmax(preds)
    else:
        preds = np.asarray(preds).astype('float64')
        exp_preds = np.exp(np.log(preds) / temperature)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)



def n_models(listof_modeltype, vocab_size, embedding_size, pretrained_weights):
    listof_model = []
    for i in range (len(listof_modeltype)):
        model = model_init(listof_modeltype[i], vocab_size, embedding_size, pretrained_weights)
        listof_model.append(model)
    return listof_model


def training(train_x, train_y, listof_model, listof_bs, epoch):
    assert len(listof_model) == len(listof_bs), "length of listof_model and listof_bs must be equal!"
    historys = []
    trained_models = []
    n = 0
    for model in listof_model:
        history = model.fit(train_x,train_y, batch_size = listof_bs[n], epochs = epoch)
        #model.fit(train_x,train_y, batch_size = listof_bs[n], epochs = epoch)
        trained_models.append(model)
        historys.append(history)
        n = n + 1
    training_graph = plot_training(historys, 'loss')
    return training_graph, trained_models


import matplotlib.pyplot as plt
def plot_training (listof_history, measurement):
    for history in listof_history:
        plt.plot(history.history[measurement])
    plt.title = ('Model LOSS')
    plt.xlabel('epoch')
    plt.ylabel('loss amount')
    return plt.show()


lsmodel = n_models(['bidirectional', 'uni'], vocab_size, embedding_size, pretrained_weights )

training_graph, trained_models = training(train_x, train_y, lsmodel, [64, 32], 100)


for model, i in enumerate (trained_models):
    model.save("textmodel" + str(i) + '.h5')

def prediction (text, listof_models, word_model, num_word_gen, temperature,n):
    sentences = []
    for model in listof_models:
        sentences.append('GENERATING SENTENCE')
        sent = [generate_next(text, word_model, model, num_word_gen, temperature) for i in range (n)]
        sentences.append(sent)
        sentences.append('FINISHED')
    return sentences



prediction('', trained_models, word_model, 5, 0.9, 5)
