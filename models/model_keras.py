import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, BatchNormalization ,Reshape
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Layer
import tensorflow.keras.backend as K

class SelfAttention(Layer):
    def __init__(self, 
                 n_hop,
                 hidden_dim,
                 nc=256,
                 penalty=1.0,
                 return_attention=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        self.n_hop = n_hop
        self.hidden_dim = hidden_dim
        self.nc=nc
        self.penalty = penalty
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.return_attention = return_attention
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (None, Sequence_size, Sequence_hidden_dim)
        assert len(input_shape) >= 3
        batch_size, T, nh = input_shape
        
        self.Ws1 = self.add_weight(shape=(self.hidden_dim, self.nc),
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws1',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        self.Ws2 = self.add_weight(shape=(self.nc, self.n_hop), 
                                      initializer=self.kernel_initializer,
                                      name='SelfAttention-Ws2',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        super(SelfAttention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 3
        assert input_shape[-1]
        batch_size, sequence_size, sequence_hidden_dim = input_shape
        output_shape = tuple([batch_size, self.n_hop, sequence_hidden_dim])
        
        if self.return_attention:
            attention_shape = tuple([batch_size, self.n_hop, sequence_size])
            return [output_shape, attention_shape]
        else: return output_shape



    
    def _frobenius_norm(self, inputs):
        outputs = K.sqrt(K.sum(K.square(inputs)))
        return outputs    

    def call(self, inputs):
        shape=inputs.shape
        H=inputs
        x = K.tanh(tf.matmul(H,self.Ws1))
        x = tf.matmul(x,self.Ws2)
        A = K.softmax(x,axis=0) # A = softmax(dot(Ws2, d1))
        At=K.permute_dimensions(A,(0,2,1))
        E = tf.matmul(At,H)
        
        return E
   
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_hop': self.n_hop,
            'hidden_dim': self.hidden_dim,
            'nc': self.nc,
            'penalty': self.penalty,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'return_attention': self.return_attention,
        })
        return config


def vgg_att(n_class):
    inputs = keras.Input(shape=(300,40,1))
    x=Conv2D(64, (3, 3), padding='same', name='block1_conv1',activation='relu')(inputs)
    x=Conv2D(64, (3, 3), padding='same', name='block1_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    print(x.shape)

    x=Conv2D(128, (3, 3), padding='same', name='block2_conv1',activation='relu')(x)
    x=Conv2D(128, (3, 3), padding='same', name='block2_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2))(x)
    print(x.shape)


    x=Conv2D(256, (3, 3), padding='same', name='block3_conv1',activation='relu')(x)
    x=Conv2D(256, (3, 3), padding='same', name='block3_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2),padding="same")(x)
    print(x.shape)

    x=Conv2D(512, (3, 3), padding='same', name='block4_conv1',activation='relu')(x)
    x=Conv2D(512, (3, 3), padding='same', name='block4_conv2',activation='relu')(x)
    x=BatchNormalization()(x)
    x=MaxPooling2D(pool_size = (2, 2), strides = (2, 2),padding="same")(x)
    print(x.shape)

    att=SelfAttention(n_hop=4,hidden_dim=1536)
    x=Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    print("after reshape")
    print(x.shape)
    x=att(x)
    print("after attention")
    print(x.shape)
    x=AveragePooling1D(pool_size=4,data_format="channels_last")(x)
    print("after avgpool")
    print(x.shape)
    x = Flatten()(x)
    x = Dense(256, activation = 'relu')(x)
    output = Dense(n_class,activation = 'softmax')(x)
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(loss='categorical_crossentropy',optimizer ='adam')#need hyperparam-tuning 
    model.summary()
    return model








