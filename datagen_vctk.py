#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.utils import Sequence, to_categorical
from load_vctk import get_model_data
import math
import numpy as np
import os


data_main_dir = os.path.join('..', 'datasets', 'vctk', 'wav48_silence_trimmed')

class VCTKDatagen(Sequence):
    def __init__(self, audio_paths, labels, batch_size, num_class, audio_load_func, shuffle=False):
        self.aud_paths = audio_paths
        self.labels = labels
        self.b_size = batch_size
        self.num_class = num_class
        self.audio_load_func = audio_load_func
        self.shuffle = audio_load_func
        
    def __len__(self):
        return math.ceil( len( self.aud_paths) / self.b_size )
    
    def __getitem__(self, idx):
        
        # Get portion of data for batch
        batch_paths = self.aud_paths[idx*self.b_size:(idx+1)*self.b_size]
        batch_labels = self.labels[idx*self.b_size:(idx+1)*self.b_size]
        
        model_in = np.array([self.audio_load_func(ap) for ap in batch_paths])
        model_out = to_categorical(batch_labels, num_classes=self.num_class)
        
        return np.expand_dims(model_in, axis=-1), model_out
    
    def on_epoch_end(self):
        if self.shuffle:
            idx = np.arange(len(self.aud_paths))
            np.random.shuffle(idx)
            self.aud_paths = np.array(self.aud_paths)[idx].tolist()
            self.labels = np.array(self.labels)[idx].tolist()

def get_datagen(sample_per_person, batch_size, audio_load_func, split=[0.1, 0.1], shuffle=True, mics=[1, 2]):
    """
    Get datagens for vctk dataset. 
    Params:
        sample_per_person: Number of samples to select for each person.
        batch_size: Batch size of the model
        audio_load_func: Function to use audio files
        split: Ratios for the test and validation sets. Default values are 0.1 for test and 0.1 for validation.
        shuffle: Whether to shuffle the paths and labels before returning them. If you pass this false, consecutive audio files
        will obtanied from same person.
        mics: Mic number of the selected audio samples. Can be one of [1], [2], [1, 2]. If both mics included
        The code could return same audio files recorded from both mics. 
    Returns:
        Datagens for train, validation and test sets
    """
    [tr_aud, tr_label], [val_aud, val_label], [te_aud, te_label] = get_model_data(data_main_dir , sample_per_person, split, shuffle, mics)
    
    # -2 for s5 and log.txt files
    n_person = len(os.listdir(data_main_dir)) - 2
    tr_gen = VCTKDatagen(tr_aud, tr_label, batch_size, n_person, audio_load_func, shuffle)
    val_gen = VCTKDatagen(val_aud, val_label, batch_size, n_person, audio_load_func, shuffle)
    te_gen = VCTKDatagen(te_aud, te_label, batch_size, n_person, audio_load_func, shuffle)
    
    return tr_gen, val_gen, te_gen
