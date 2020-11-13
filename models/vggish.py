from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation, BatchNormalization
from tensorflow.keras import backend as K

model_path = '/home/bbekci/inzpeech/models/vggish_audioset_weights_without_fc2.h5'

def VGGish(input_shape, num_classes, add_output=True, load_weight=False):

    aud_input = Input(shape=input_shape, name='input_1')
    
    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv1')(aud_input)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv3/conv3_1')(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv3/conv3_2')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv4/conv4_1')(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), activation=None, padding='same', name='conv4/conv4_2')(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    base_model = Model(aud_input, x)
    base_model.load_weights(model_path)

    if load_weight:
        base_model.load_weights(model_path)

    x = GlobalMaxPooling2D()(base_model.output)
    x = Dense(4096, activation=None, name='vggish_fc1/fc1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(4096, activation=None, name='vggish_fc1/fc1_2')(x)
    x = BatchNormalization()(x)
    preds = Dense(num_classes, activation='softmax', name='vggish_fc2')(x)
    
    if add_output:
        model = Model(aud_input, preds, name='VGGish')
    else:
        model = Model(aud_input, x, name='VGGish')

    return model