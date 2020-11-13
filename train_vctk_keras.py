#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models.vggish import VGGish
from datagen_vctk import get_datagen
from utils import apply_melspectrogram_to_file
import math
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


sample_per_person = 30000
batch_size = 64
num_class = 109
input_shape = (300, 40, 1)


tr_gen, val_gen, te_gen = get_datagen(sample_per_person, batch_size, apply_melspectrogram_to_file)

for x, y in tr_gen:
    print(x.shape)
    print(y.shape)
    break

reduceLR = ReduceLROnPlateau(factor=0.5, patience=5, verbose=True)
earlystop = EarlyStopping(patience=15, verbose=True)
model = VGGish(input_shape, num_class)

opt = Adam(lr=2e-3)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(tr_gen, validation_data=val_gen, epochs=15, callbacks=[reduceLR])

