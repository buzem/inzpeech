#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from tensorflow.keras.utils import Sequence, to_categorical
import math
import random
from dataloaders.datautil import get_pkl_paths

class DataVoxCeleb():

    def __init__(self, data_dir, pkl_paths, feature_len=300):
        self.ids = [fname for fname in os.listdir(data_dir) if 'id' in fname]
        self.id_2_labels = { k:i for i, k in enumerate(self.ids)}
        self.feature_len = feature_len
        self.pkl_paths = pkl_paths
        self.shuffle_set()

    def shuffle_set(self):
        np.random.shuffle(self.pkl_paths)
    
    def get_num_ex(self):
        return len(self.pkl_paths)
    
    def get_n_class(self):
        return len(self.id_2_labels.keys())

    def get_name_to_label(self, name):
        return int(self.id_2_labels[name])

    def get_sample(self, idx):
        sample_pickle_path = self.pkl_paths[idx]
        with open(sample_pickle_path, 'rb') as pickle_load:
            loaded_sample = pickle.load(pickle_load)
        
        idname, videoname, features = loaded_sample

        feature_len = features.shape[0]
        upper_limit = feature_len - 300
        feature_start = random.randint(0, upper_limit)

        return features[feature_start:(feature_start+self.feature_len), :], self.get_name_to_label(idname)
        
    def get_batch_sample(self, idx, batch_size):
        
        sample_pickle_path = self.pkl_paths[idx*batch_size:(idx+1)*batch_size]
        num_example = len(sample_pickle_path)
        batch_features = np.zeros((num_example, self.feature_len, 40))
        batch_labels = np.zeros(num_example, dtype=np.int)
        
        for i, pp in enumerate(sample_pickle_path):
            with open(pp, 'rb') as pickle_load:
                loaded_sample = pickle.load(pickle_load)
            
            idname, videoname, features = loaded_sample
            
            feature_len = features.shape[0]
            upper_limit = feature_len - 300
            feature_start = random.randint(0, upper_limit)

            batch_features[i] = features[feature_start:(feature_start+self.feature_len), :].copy()
            batch_labels[i] = self.get_name_to_label(idname)
        
        return batch_features, batch_labels

class DataTorchVoxCeleb(Dataset):
    
    def __init__(self, file_dir, pkl_paths, feature_len):
        self.datagen = DataVoxCeleb(file_dir, pkl_paths, feature_len)

    def __len__(self):
        return self.datagen.get_num_ex()

    def n_class(self):
        return self.datagen.get_n_class()

    def __getitem__(self, idx):
        data, label =  self.datagen.get_sample(idx)
        data = np.expand_dims(data, axis=0)
        return data, label

class DataKerasVoxCeleb(Sequence):
    
    def __init__(self, file_dir, pkl_paths, feature_len, batch_size, shuffle=False):
        self.datagen = DataVoxCeleb(file_dir, pkl_paths, feature_len)
        self.shuffle = shuffle
        self.batch_size = batch_size
        
    def __len__(self):
        return math.ceil(self.datagen.get_num_ex() / self.batch_size)

    def n_class(self):
        return self.datagen.get_n_class()

    def __getitem__(self, idx):
        #return self.datagen.get_batch_sample(idx, self.batch_size)
        data, label = self.datagen.get_batch_sample(idx, self.batch_size)
        return data, to_categorical(label, num_classes=self.n_class())

    def on_epoch_end(self):
        if self.shuffle == True:
            self.datagen.shuffle_set()


def get_torch_datagens(data_dir, feature_len=300, num_video_per_person=1e4, num_audio_per_video=1e4, split_by='audio', split_size=0.2, txt_dirs=None, ratios=[1.0, 1.0]):
    """
    Returns datagens for torch
    Params:
        data_dir: Parent directory for the pickle files. Assumed that each person has a separate folder in data_dir and each
        video has separate folder in each persons' folder. Lastly pickle files included in video folders.
        feature_len: Number of features samples for each audio sample. Deafult is 300.
        num_video_per_person: How many videos will be selected from each person folder
        num_audio_per_video: How many audio files will be selected from each video folder
        split_by: One of "video" or "audio". If "video" provided than audio files from a single video will be included in either
        train or validation set. If the parameter passed as "audio", all pickle files will be splitted into train and validation. 
        txt_dirs: Directory for the train and validation text files. Pass as [train_file_path, validation_file_path]
        ratios: Ratio of the subsets.
    Returns:
        train and validation pickle paths
    """
    tr_paths, val_paths = get_pkl_paths(data_dir, num_video_per_person, num_audio_per_video, split_by, split_size, txt_dirs)

    subset_tr = np.random.choice(tr_paths, size=math.ceil(len(tr_paths) * ratios[0]), replace=False)
    subset_val = np.random.choice(val_paths, size=math.ceil(len(val_paths) * ratios[1]), replace=False)

    tr_gen = DataTorchVoxCeleb(data_dir, subset_tr, feature_len)
    val_gen = DataTorchVoxCeleb(data_dir, subset_val, feature_len)
    
    return tr_gen, val_gen

def get_keras_datagens(data_dir, batch_size, feature_len=300, num_video_per_person=1e4, num_audio_per_video=1e4, split_by='audio', split_size=0.2, txt_dirs=None, ratios=[1.0, 1.0]):
    """
    Returns datagens for keras
    Params:
        data_dir: Parent directory for the pickle files. Assumed that each person has a separate folder in data_dir and each
        video has separate folder in each persons' folder. Lastly pickle files included in video folders.
        batch_size: Batch size.
        feature_len: Number of features samples for each audio sample. Deafult is 300.
        num_video_per_person: How many videos will be selected from each person folder
        num_audio_per_video: How many audio files will be selected from each video folder
        split_by: One of "video" or "audio". If "video" provided than audio files from a single video will be included in either
        train or validation set. If the parameter passed as "audio", all pickle files will be splitted into train and validation. 
        txt_dirs: Directory for the train and validation text files. Pass as [train_file_path, validation_file_path]
        ratios: Ratio of the subsets.
    Returns:
        train and validation pickle paths
    """
    tr_paths, val_paths = get_pkl_paths(data_dir, num_video_per_person, num_audio_per_video, split_by, split_size, txt_dirs)

    subset_tr = np.random.choice(tr_paths, size=math.ceil(len(tr_paths) * ratios[0]), replace=False)
    subset_val = np.random.choice(val_paths, size=math.ceil(len(val_paths) * ratios[1]), replace=False)
    
    tr_gen = DataKerasVoxCeleb(data_dir, subset_tr, feature_len, batch_size, True)
    val_gen = DataKerasVoxCeleb(data_dir, subset_val, feature_len, batch_size, False)

    return tr_gen, val_gen
