from keras.layers import *
from keras.models import Model, load_model

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers #keras2

import numpy as np
from keras.optimizers import *
import keras

from Utils import *

def get_image_model(lr):
    
    image_input = Input(shape=(28,28,1),)

    image_rep = Conv2D(32,(5,5),)(image_input)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Conv2D(128,(5,5),)(image_rep)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Flatten()(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    logit = Dense(62,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr),
                      metrics=['acc'])

    return model



class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')
    user_vecs =Dropout(0.2)(vecs_input)
    user_att = keras.layers.Flatten()(Dense(1,use_bias=False)(user_vecs))
    user_att = Activation('softmax')(user_att)
    user_vec = keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model


def get_model(model_mode,lr,max_length,cate_num):
    
    sentence_input = Input(shape=(max_length,300),)
    
    title_vecs = sentence_input
    if model_mode == 'Trans':
        title_vecs = Attention(20,20)([title_vecs]*3)
    elif model_mode == 'CNN':
        title_vecs = Conv1D(400,3,padding='same')(title_vecs)
    elif model_mode == 'BiLSTM':
        title_vecs = Bidirectional(LSTM(200))(title_vecs)
        
    title_vecs = Dropout(0.2)(title_vecs)
    title_vec = AttentivePooling(30,400)(title_vecs)
    
    vec = Dense(256,activation='relu')(title_vec)
    vec = Dense(256,activation='relu')(vec)
    logit = Dense(cate_num,activation='softmax')(vec)
    
    model = Model(sentence_input,logit) # max prob_click_positive
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr),
                      metrics=['acc'])

    return model