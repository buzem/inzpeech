#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from sklearn.model_selection import train_test_split

def get_person_label(pname):
    return int(pname.replace('p', ''))

def get_samples_from_person(person_path, sample_count, mics):
    """
    Return path of audio samples selected from a person folder.
    Params:
        person_path: Path for the person
        sample_count: Number of samples to select
        mics: Mic number of the selected audio samples. Can be one of [1], [2], [1, 2]. If both mics included
        The code could return same audio files recorded from both mics. 
    Returns:
        audio_paths: Relative path of the audio samples
    """
    audio_files = os.listdir(person_path)
    mic_string = ['mic'+ str(n) for n in mics ]
    audio_files = [af for af in audio_files if af.split('.')[0].split('_')[-1] in mic_string]
    sample_count = min(len(audio_files), sample_count)
    audio_paths = [os.path.join(person_path, af) for af in audio_files]
    return np.random.choice(audio_paths, sample_count, replace=False).tolist()

def get_model_data(data_main_dir, sample_per_person, split=[0.1, 0.1], shuffle=True, mics=[1,2]):
    """
    Return audio file paths and corresponding labels.
    Params:
        data_main_dir: Parent directory for the dataset
        sample_per_person: Number of samples to select
        split: Ratios for the test and validation sets. Default values are 0.1 for test and 0.1 for validation.
        shuffle: Whether to shuffle the paths and labels before returning them. If you pass this false, consecutive audio files
        will obtanied from same person.
        mics: Mic number of the selected audio samples. Can be one of [1], [2], [1, 2]. If both mics included
        The code could return same audio files recorded from both mics. 
    Returns:
        audio paths and labels for each subset. Audio paths and labels are given as a single list for each subset
    """
    all_audio_paths = []
    labels = []
    
    person_names = [pname for pname in os.listdir(data_main_dir) if 'p' in pname]
    person_paths = [os.path.join(data_main_dir, p) for p in person_names]

    for i, ppath in enumerate(person_paths):
            audio_paths = get_samples_from_person(ppath, sample_per_person, mics)
            labels = labels + len(audio_paths) * [i]
            all_audio_paths = all_audio_paths + audio_paths
    
    if shuffle:
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        
        labels = np.array(labels)[idx].tolist()
        all_audio_paths = np.array(all_audio_paths)[idx].tolist()
        
    tr_val_audio, test_audio, tr_val_labels, te_labels = train_test_split(all_audio_paths, labels, test_size=split[0], random_state=42)
    
    tr_audio, val_audio, tr_labels, val_labels = train_test_split(tr_val_audio, tr_val_labels, test_size=split[1], random_state=42)
    
    return [tr_audio, tr_labels], [val_audio, val_labels], [test_audio, te_labels]

def get_model_data_for_batch(data_main_dir, sample_per_person, person_count_per_batch , shuffle=True, mics=[1,2]):
    """
    Return audio file paths and corresponding labels for a batch.
    Params:
        data_main_dir: Parent directory for the dataset
        sample_per_person: Number of samples to select
        person_count_per_batch: Number of persons to be added for each batch. Note that the batch number will be equal to 
        sample_per_person * person_count_per_batch
        shuffle: Whether to shuffle the paths and labels before returning them. If you pass this false, consecutive audio files
        will obtanied from same person.
        mics: Mic number of the selected audio samples. Can be one of [1], [2], [1, 2]. If both mics included
        The code could return same audio files recorded from both mics. 
    Returns:
        audio_paths: Relative path of the audio samples
    """    
    all_audio_paths = []
    labels = []
    
    person_names = [pname for pname in os.listdir(data_main_dir) if 'p' in pname]
    person_paths = [os.path.join(data_main_dir, p) for p in person_names]
    person_labels = [get_person_label(pname) for pname in person_names]

    # Sample persons
    idx = np.arange(len(person_paths))
    selected_idx = np.random.choice(idx, person_count_per_batch, replace=False)
    # Select person names, paths and corresponding labels
    person_names = np.array(person_names)[selected_idx].tolist()
    person_paths = np.array(person_paths)[selected_idx].tolist()
    
    for i, ppath in enumerate(person_paths):
            audio_paths = get_samples_from_person(ppath, sample_per_person, mics)
            labels = labels + len(audio_paths) * [i]
            all_audio_paths = all_audio_paths + audio_paths
    
    if shuffle:
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        
        labels = np.array(labels)[idx].tolist()
        all_audio_paths = np.array(all_audio_paths)[idx].tolist()
    return all_audio_paths, labels

