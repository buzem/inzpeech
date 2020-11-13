
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
from dataloaders.DatagenVoxCeleb import get_keras_datagens
#from dataloaders.DatagenVoxCeleb1 import get_keras_datagens
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models.model_keras_dropout import vgg_att



# FOR VOXCELEB2
data_dir = '/media/data/bbekci/voxceleb2/data/dev/pkls/'
tr_txt = 'txts/tr_voxceleb_audio_pkl_paths.txt'
val_txt = 'txts/val_voxceleb_audio_pkl_paths.txt' 

# FOR VOXCELEB
#txt_dir = '/media/data/bbekci/voxceleb/iden_split.txt'
#data_dir = '/media/data/bbekci/voxceleb/pkls/'
batch_size = 128
input_shape = (300, 40, 1)

# VOX CELEB
#tr_gen, val_gen, te_gen = get_keras_datagens(data_dir, txt_dir, batch_size, feature_len=300, ratios=[0.05, 0.1, 0.1])

# VOX CELEB2
tr_gen, val_gen = get_keras_datagens(data_dir, batch_size, txt_dirs=[tr_txt, val_txt], ratios=[0.25, 0.1])
"""
for x,y in tr_gen:
    print(x.shape)
    print(y.shape)
    print(y)
    break

"""
n_class = tr_gen.datagen.get_n_class()
print(n_class)
vgg_base_model = vgg_att(n_class)

opt = Adam(lr=2e-4)
vgg_base_model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

save_dir = os.path.join('saved-models', 'vggish_attention_voxceleb2_dropout.h5')
check = ModelCheckpoint(save_dir, verbose=True, save_best_only=True)
reduceLR = ReduceLROnPlateau(factor=0.5, patience=3, verbose=True)
earlyStop = EarlyStopping(patience=15, verbose=True)

vgg_base_model.evaluate(val_gen)
history = vgg_base_model.fit(tr_gen, epochs=45, validation_data=val_gen, callbacks=[check, reduceLR, earlyStop])