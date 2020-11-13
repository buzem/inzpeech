
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from models.vggish import VGGish
import math
import numpy as np
import tensorflow_addons as tfa
from dataloaders.DatagenVoxCeleb1 import get_keras_datagens
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models.model_keras_dropout import vgg_att
from models.vggish import VGGish
import pickle
import tensorflow as tf
from models.resnet18_keras import resnet18

def focal_loss(y_true, y_pred):
    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    gamma=2.0
    alpha=0.25
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    #y_pred = y_pred + epsilon
    # Clip the prediction value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate cross entropy
    cross_entropy = -y_true*K.log(y_pred)
    # Calculate weight that consists of  modulating factor and weighting factor
    weight = alpha * y_true * K.pow((1-y_pred), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)
    return loss
    
# FOR VOXCELEB
txt_dir = '/media/data/bbekci/voxceleb/iden_split.txt'
data_dir = '/media/data/bbekci/voxceleb/pkls_colwise_normed/'
batch_size = 128
input_shape = (300, 40, 1)

# VOX CELEB
tr_gen, val_gen, te_gen = get_keras_datagens(data_dir, txt_dir, batch_size, feature_len=300, ratios=[1.0, 1.0, 1.0], vid_per_person=200000)

n_class = tr_gen.datagen.get_n_class()

vgg_base_model = vgg_att(n_class)
#vgg_base_model = VGGish(input_shape, n_class, load_weight=True)

#resnet_model = resnet18(n_class, True)

opt = Adam(lr=2e-4)
vgg_base_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

save_dir = os.path.join('saved-models', 'voxceleb1_attention_vgg_dropout_tfkeras_fullset.h5')
check = ModelCheckpoint(save_dir, verbose=True, save_best_only=True, monitor='val_accuracy')
reduceLR = ReduceLROnPlateau(factor=0.5, patience=10, verbose=True, monitor='val_accuracy')
earlyStop = EarlyStopping(patience=50, verbose=True, monitor='val_accuracy')

history = vgg_base_model.fit(tr_gen, epochs=3000, validation_data=val_gen, callbacks=[check, reduceLR, earlyStop])

te_loss, te_acc = vgg_base_model.evaluate(te_gen)
with open('saved-models/voxceleb1_tfkeras_vgg_att_dropout_full', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

print("Test Loss: ", te_loss, " test acc: ", te_acc)
