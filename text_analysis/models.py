'''
Created on Dec 8, 2015

@author: donghyun
'''
import keras
import numpy as np
from keras import Input
from keras.layers import GlobalMaxPooling1D
from keras.models import Model

np.random.seed(1337)

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D, Conv1D
from keras.layers.core import Reshape, Flatten, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


class CNN_module():
    '''
    classdocs
    '''
    batch_size = 128
    # More than this epoch cause easily over-fitting on our data sets
    nb_epoch = 5

    def __init__(self, output_dimesion, vocab_size, dropout_rate, emb_dim, max_len, nb_filters, init_W=None):

        self.max_len = max_len
        max_features = vocab_size
        vanila_dimension = 200
        projection_dimension = output_dimesion

        filter_lengths = [3, 4, 5]

        '''Embedding Layer'''
        inputs = Input(name='input', shape=(max_len,), dtype='int32')

        if init_W is None:
            embed_x = Embedding(max_features, emb_dim, input_length=max_len, name='sentence_embeddings')(inputs)
        else:
            embed_x = Embedding(max_features, emb_dim, input_length=max_len, weights=[init_W / 20],name='sentence_embeddings')(inputs)
        conv_list = []
        '''Convolution Layer & Max Pooling Layer'''
        for i in filter_lengths:
            # embed_x = Reshape(target_shape=(1, self.max_len, emb_dim), input_shape=(self.max_len, emb_dim))(embed_x)
            feature_map = Conv1D(nb_filters, i, activation="relu",name="conv_"+str(i))(embed_x)
            max_x = GlobalMaxPooling1D(name="max_pooling_"+str(i))(feature_map)
            # max_x = Flatten()(max_x)
            conv_list.append( max_x )
        ## concat all conv output vector into a long vector
        sentence_vector = keras.layers.concatenate(conv_list)
        '''Dropout Layer'''
        fc1_x = Dense(vanila_dimension, activation='tanh',name='fully_connect')(sentence_vector)
        drop_x = Dropout(dropout_rate,name='dropout')(fc1_x)
        '''Projection Layer & Output Layer'''
        out = Dense(projection_dimension, activation='tanh',name='projection')(drop_x)

        # Output Layer
        self.model = Model(inputs=inputs, outputs=out)
        self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mse'])

    def load_model(self, model_path):
        self.model.load_weights(model_path)

    def save_model(self, model_path, isoverwrite=True):
        self.model.save_weights(model_path, isoverwrite)

    # def qualitative_CNN(self, vocab_size, emb_dim, max_len, nb_filters):
    #     self.max_len = max_len
    #     max_features = vocab_size
    #
    #     filter_lengths = [3, 4, 5]
    #     print("Build model...")
    #     self.qual_conv_set = {}
    #     '''Embedding Layer'''
    #     Input(name='input', shape=(max_len,), dtype=int)
    #
    #     self.qual_model.add_node(Embedding(max_features, emb_dim, input_length=max_len, weights=self.model.nodes['sentence_embeddings'].get_weights()),
    #                              name='sentence_embeddings', input='input')
    #
    #     '''Convolution Layer & Max Pooling Layer'''
    #     for i in filter_lengths:
    #         model_internal = Sequential()
    #         model_internal.add(
    #             Reshape(dims=(1, max_len, emb_dim), input_shape=(max_len, emb_dim)))
    #         self.qual_conv_set[i] = Convolution2D(nb_filters, i, emb_dim, activation="relu", weights=self.model.nodes[
    #                                               'unit_' + str(i)].layers[1].get_weights())
    #         model_internal.add(self.qual_conv_set[i])
    #         model_internal.add(MaxPooling2D(pool_size=(max_len - i + 1, 1)))
    #         model_internal.add(Flatten())
    #
    #         self.qual_model.add_node(
    #             model_internal, name='unit_' + str(i), input='sentence_embeddings')
    #         self.qual_model.add_output(
    #             name='output_' + str(i), input='unit_' + str(i))
    #     self.qual_model = Graph()
    #     self.qual_model.compile(
    #         'rmsprop', {'output_3': 'mse', 'output_4': 'mse', 'output_5': 'mse'})

    def train(self, X_train, V, item_weight, seed):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        np.random.seed(seed)
        X_train = np.random.permutation(X_train)
        np.random.seed(seed)
        V = np.random.permutation(V)
        np.random.seed(seed)
        item_weight = np.random.permutation(item_weight)

        print("Train...CNN module")
        # print('x={X_train}, y={V},verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch, sample_weight={item_weight}'.format(X_train=X_train,V=V,item_weight=item_weight ))
        history = self.model.fit(x=X_train, y=V,verbose=0, batch_size=self.batch_size, epochs=self.nb_epoch, sample_weight=item_weight)

        # cnn_loss_his = history.history['loss']
        # cmp_cnn_loss = sorted(cnn_loss_his)[::-1]
        # if cnn_loss_his != cmp_cnn_loss:
        #     self.nb_epoch = 1
        return history

    def get_projection_layer(self, X_train):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_len)
        Y = self.model.predict(X_train, batch_size=len(X_train))
        print('output:%s' %Y)
        return Y