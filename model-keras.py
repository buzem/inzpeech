import keras
from keras_self_attention import SeqSelfAttention
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Import
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation 
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

class SelfAttention(Layer):
    def __init__(self, 
                 n_hop,
                 hidden_dim,
                 penalty=1.0,
                 return_attention=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.n_hop = n_hop
        self.hidden_dim = hidden_dim
        self.penalty = penalty
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.return_attention = return_attention
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, Sequence_size, Sequence_hidden_dim)
        assert len(input_shape) >= 3
        batch_size, sequence_size, sequence_hidden_dim = input_shape
        
        self.Ws1 = self.add_weight(shape=(self.hidden_dim, sequence_hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.Ws2 = self.add_weight(shape=(self.n_hop, self.hidden_dim), 
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        batch_size = K.cast(K.shape(inputs)[0], K.floatx())
        inputs_t = K.permute_dimensions(inputs, (1,2,0)) # H.T
        d1 = K.tanh(K.permute_dimensions(K.dot(self.Ws1, inputs_t), (2,0,1))) # d1 = tanh(dot(Ws1, H.T))
        d1 = K.permute_dimensions(d1, (2,1,0))
        A = K.softmax(K.permute_dimensions(K.dot(self.Ws2, d1), (2,0,1))) # A = softmax(dot(Ws2, d1))
        H = K.permute_dimensions(inputs, (0,2,1))
        outputs = K.batch_dot(A, H, axes=2) # M = AH

        A_t = K.permute_dimensions(A, (0,2,1))
        I = K.eye(self.n_hop)
        P = K.square(self._frobenius_norm(K.batch_dot(A, A_t) - I)) # P = (frobenius_norm(dot(A, A.T) - I))**2
        self.add_loss(self.penalty*(P/batch_size))
        
        if self.return_attention: 
            return [outputs, A]
        else: 
            return outputs

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 3
        assert input_shape[-1]
        batch_size, sequence_size, sequence_hidden_dim = input_shape
        output_shape = tuple([batch_size, self.n_hop, sequence_hidden_dim])
        
        if self.return_attention:
            attention_shape = tuple([batch_size, self.n_hop, sequence_size])
            return [output_shape, attention_shape]
        else: return output_shape


    def get_config(self):
        config = {
            'n_hop': self.n_hop,
            'hidden_dim': self.hidden_dim,
            'penalty':self.penalty,
            'return_attention': self.return_attention,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _frobenius_norm(self, inputs):
        outputs = K.sqrt(K.sum(K.square(inputs)))
        return outputs



def vgg_att():
    inputs = keras.Input(shape=(300,40,))
    x=Conv2D(64, (3, 3), padding='same', name='block1_conv1',activation='relu')(inputs)
    x=Conv2D(64, (3, 3), padding='same', name='block1_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    x=Conv2D(128, (3, 3), padding='same', name='block2_conv1',activation='relu')(x)
    x=Conv2D(128, (3, 3), padding='same', name='block2_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)


    x=Conv2D(256, (3, 3), padding='same', name='block3_conv1',activation='relu')(x)
    x=Conv2D(256, (3, 3), padding='same', name='block3_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    x=Conv2D(512, (3, 3), padding='same', name='block4_conv1',activation='relu')(x)
    x=Conv2D(512, (3, 3), padding='same', name='block4_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)

    att=SelfAttention(n_hop=4,hidden_dim=1536)
    x=att(x)
    x=AveragePooling2D(x,pool_size=(4, 1))
    x = Flatten()(x)
    x = Dense(256, activation = 'relu')(x)
    output = Dense(1251,activation = 'softmax')(x)
    model = Model(inputs=inputs, outputs=output)

    model.compile(optimizer = Adam(lr=0.1))#need hyperparam-tuning 
    model.summary()
    return model








