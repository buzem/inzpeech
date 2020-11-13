import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling1D, BatchNormalization ,Reshape, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Layer, Add, Input, GlobalAveragePooling2D
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
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
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


def resnet_block(input_tensor, kernel_size, filters, downsample):

    first_stride = 1
    if downsample:
        first_stride = 2
    # First Block
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding="same", strides=first_stride)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)


    # Second Block
    x = Conv2D(kernel_size=kernel_size, filters=filters, padding="same", strides=1)(x)
    x = BatchNormalization()(x)

    
    input_tensor = Conv2D(kernel_size=(1,1), filters=filters,  padding='same', strides=first_stride)(input_tensor)
    
    # Final Add Layer
    x = Add()([input_tensor, x])
    x = Activation("relu")(x)

    return x

def resnet18(n_class, add_attention):

    input_layer = Input(shape=(300,40,1))
    x=Conv2D(32, (7, 7), padding='same' ,strides=1)(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    x = Dropout(0.15)(x)

    x = resnet_block(x, 3, 32, True)
    x = resnet_block(x, 3, 32, False)
    x = Dropout(0.15)(x)

    x = resnet_block(x, 3, 64, True)
    x = resnet_block(x, 3, 64, False)
    x = Dropout(0.15)(x)

    x = resnet_block(x, 3, 128, True)
    x = resnet_block(x, 3, 128, False)
    x = Dropout(0.15)(x)

    x = resnet_block(x, 3, 256, True)
    x = resnet_block(x, 3, 256, False)
    x = Dropout(0.15)(x)

    #  Attention here
    if add_attention:
        att=SelfAttention(n_hop=4,hidden_dim=512)
        x=Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
        x=att(x)
        x=AveragePooling1D(pool_size=4,data_format="channels_last")(x)
        x = Flatten()(x)
    
    else:
        x = GlobalAveragePooling2D()(x)
  
    x = Dropout(0.15)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    preds = Dense(1251, activation='softmax')(x)

    model = Model(input_layer, preds)
    model.summary()
    return model



