import os
import math
import numpy as np
from dataloaders.DatagenVoxCeleb import DataTorchVoxCeleb, DataKerasVoxCeleb

class Video:
    def __init__(self, name):
        self.name = name
        self.audios = []

    def add_audio(self, audio_path):
        self.audios.append(audio_path)        

    def __eq__(self, other):
        if len(self.audios)==len(other.audios):
                return True
        return False
        
    def __lt__(self, other):
        if len(self.audios) < len(other.audios):
            return True
        return False
class ID:
    def __init__(self, name):
        self.name = name
        self.videos = []
    
    def add_audio(self, video_name, audio_path):
        vid_found = False
        for v in self.videos:
            if v.name == video_name:
                vid_found = True
                v.add_audio(audio_path)
                break
        if not vid_found:
            v = Video(video_name)
            v.add_audio(audio_path)
            self.videos.append(v)
    
    def get_audio_count(self):
        count = 0 
        for v in self.videos:
            count += len(v.audios)
        return count

    def get_person_audio_paths(self):
        paths = []
        for v in self.videos:
            paths.extend(v.audios)
        return paths


class Dataset:
    def __init__(self):
        self.ids = []
    
    def add_audio(self, id_name, video_name, audio_path):
        id_found = False
        for i in self.ids:
            if i.name == id_name:
                id_found = True
                i.add_audio(video_name, audio_path)
                break
        if not id_found:
            id = ID(id_name)
            id.add_audio(video_name, audio_path)
            self.ids.append(id)

    def get_cleaned_paths(self, vid_per_person, return_max):
        final_paths = []
        for id in self.ids:
            id.videos.sort()
            if return_max:
                for v in id.videos[-vid_per_person:]:
                    final_paths.extend(v.audios)
            else:
                for v in id.videos[:vid_per_person]:
                    final_paths.extend(v.audios)
        return final_paths

    def get_balanced_paths(self, sample_per_person):
        final_paths = []
        
        for pid in self.ids:
            person_audio_paths = pid.get_person_audio_paths()
            sampled_paths = np.random.choice(person_audio_paths, min(sample_per_person, len(person_audio_paths)), replace=False)
            final_paths.extend(sampled_paths)
        
        return final_paths

def clean_trainset(tr_paths, vid_per_person, return_max):
    trset = Dataset()

    for p in tr_paths:
        sliced_path = p.split('/')
        aud_name = sliced_path[-1]
        vid_name = sliced_path[-2]
        p_name = sliced_path[-3]
        
        trset.add_audio(p_name, vid_name, p)

    return trset.get_balanced_paths(70)



def get_voxceleb1_path(data_dir, txt_path, ratios, vid_per_person, return_max):

    with open(txt_path, 'r') as identxt:
        lines = identxt.readlines()

    train_paths = []
    test_paths = []
    val_paths = []

    for line in lines:
        subset, path = line.strip().split(' ')
        if subset == '1':
            train_paths.append(os.path.join(data_dir, path.replace('.wav','.pkl')))
        elif subset == '2':
            val_paths.append(os.path.join(data_dir, path.replace('.wav','.pkl')))
        elif subset == '3':
            test_paths.append(os.path.join(data_dir, path.replace('.wav','.pkl')))

    subset_tr = np.random.choice(train_paths, size=math.ceil(len(train_paths) * ratios[0]), replace=False)
    #subset_tr = clean_trainset(train_paths, vid_per_person, return_max)
    subset_val = np.random.choice(val_paths, size=math.ceil(len(val_paths) * ratios[1]), replace=False)
    subset_te = np.random.choice(test_paths, size=math.ceil(len(test_paths) * ratios[2]), replace=False)

    print("Original size of the training: {} size of the subset: {}".format(len(train_paths), len(subset_tr)))
    print("Original size of the validation: {} size of the subset: {}".format(len(val_paths), len(subset_val)))
    print("Original size of the testing: {} size of the subset: {}".format(len(test_paths), len(subset_te)))
    
    return subset_tr, subset_val, subset_te


def get_torch_datagens(data_dir, txt_dir, feature_len=300, ratios=[1.0, 1.0, 1.0], vid_per_person=1, return_max=True):
    """
    Returns datagens for torch
    Params:
        data_dir: Parent directory for the pickle files. Assumed that each person has a separate folder in data_dir and each
        video has separate folder in each persons' folder. Lastly pickle files included in video folders.
        txt_dir: Directory for the subset split of the dataset. 
        feature_len: Number of features samples for each audio sample. Deafult is 300.
        ratios: Ratio for splitting each train, validation and test sets. Can be used to work with smaller dataset. Default value is [1., 1., 1.] for [train, val, test]
        vid_per_person: Select samples of audios from how many videos per person.
        return_max: Whether select the videos containing max audio samples.
    Returns:
        train, validation and test datagens
    """
    tr_paths, val_paths, test_paths = get_voxceleb1_path(data_dir, txt_dir, ratios, vid_per_person, return_max)
    
    tr_gen = DataTorchVoxCeleb(data_dir, tr_paths, feature_len)
    val_gen = DataTorchVoxCeleb(data_dir, val_paths, feature_len)
    test_gen = DataTorchVoxCeleb(data_dir, test_paths, feature_len)
    
    return tr_gen, val_gen, test_gen

def get_keras_datagens(data_dir, txt_dir, batch_size, feature_len=300, ratios=[1.0, 1.0, 1.0], vid_per_person=1, return_max=True):
    """
    Returns datagens for keras
    Params:
        data_dir: Parent directory for the pickle files. Assumed that each person has a separate folder in data_dir and each
        video has separate folder in each persons' folder. Lastly pickle files included in video folders.
        txt_dir: Directory for the subset split of the dataset. 
        batch_size: Batch size to use.
        feature_len: Number of features samples for each audio sample. Deafult is 300.
        ratios: Ratio for splitting each train, validation and test sets. Can be used to work with smaller dataset. Default value is [1., 1., 1.] for [train, val, test]
        vid_per_person: Select samples of audios from how many videos per person.
        return_max: Whether select the videos containing max audio samples.
    Returns:
        train, validation and test datagens
    """
    tr_paths, val_paths, test_paths = get_voxceleb1_path(data_dir, txt_dir, ratios, vid_per_person, return_max)
    
    tr_gen = DataKerasVoxCeleb(data_dir, tr_paths, feature_len, batch_size, True)
    val_gen = DataKerasVoxCeleb(data_dir, val_paths, feature_len, batch_size, False)
    te_gen = DataKerasVoxCeleb(data_dir, test_paths, feature_len, batch_size, False)

    return tr_gen, val_gen, te_gen
