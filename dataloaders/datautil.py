import os
import numpy as np
from sklearn.model_selection import train_test_split


def get_pkl_paths(data_dir, num_video_per_person, num_audio_per_video, split_by, split_size, txt_dirs=None):
    """
    Returns pickle file paths from the given data directory.
    Params:
        data_dir: Parent directory for the pickle files. Assumed that each person has a separate folder in data_dir and each
        video has separate folder in each persons' folder. Lastly pickle files included in video folders.
        num_video_per_person: How many videos will be selected from each person folder
        num_audio_per_video: How many audio files will be selected from each video folder
        split_by: One of "video" or "audio". If "video" provided than audio files from a single video will be included in either
        train or validation set. If the parameter passed as "audio", all pickle files will be splitted into train and validation. 
        txt_dirs: Directory for the train and validation text files. Pass as [train_file_path, validation_file_path]
    Returns:
        train and validation pickle paths
    """
    tr_pkl_paths = []
    val_pkl_paths = []
    
    if txt_dirs is not None:
        with open(txt_dirs[0], 'r') as path_file:
            tr_pkl_paths = path_file.readlines()
            tr_pkl_paths = [pr.strip() for pr in tr_pkl_paths]
        with open(txt_dirs[1], 'r') as path_file:
            val_pkl_paths = path_file.readlines()
            val_pkl_paths = [pr.strip() for pr in val_pkl_paths]
            
        return tr_pkl_paths, val_pkl_paths

    ids = [fname for fname in os.listdir(data_dir) if 'id' in fname]

    for i, nid in enumerate(ids):
        print("Progress: {} / {}".format(i, len(ids)), end='\r')
        idpath = os.path.join(data_dir, nid)
        videos_names = os.listdir(idpath)
        tr_video_names = np.random.choice(videos_names, size=min(num_video_per_person, len(videos_names)), replace=False)
        
        val_video_names = []

        if split_by == 'video':
            tr_video_names, val_video_names = train_test_split(tr_video_names, random_state=42, test_size=split_size)
        
        for vname in tr_video_names:
            val_audio_names = []
                
            vidpath = os.path.join(idpath, vname)
            audio_names = os.listdir(vidpath)
            tr_audio_names = np.random.choice(audio_names, size=min(num_audio_per_video, len(audio_names)), replace=False)
            
            # There must be at least 1 audio file for validation
            if split_by == 'audio' and (len(tr_audio_names) * split_size) > 0.99:
                tr_audio_names, val_audio_names = train_test_split(tr_audio_names, random_state=42, test_size=split_size)
            
            tr_pkl_paths = tr_pkl_paths + [os.path.join(vidpath, aud_name) for aud_name in tr_audio_names if '.pkl' in aud_name]
            val_pkl_paths = val_pkl_paths + [os.path.join(vidpath, aud_name) for aud_name in val_audio_names if '.pkl' in aud_name]
            
        for vname in val_video_names:

            vidpath = os.path.join(idpath, vname)
            audio_names = os.listdir(vidpath)
            selected_audio_names = np.random.choice(audio_names, size=min(num_audio_per_video, len(audio_names)), replace=False)

            val_pkl_paths = val_pkl_paths + [os.path.join(vidpath, aud_name) for aud_name in selected_audio_names if '.pkl' in aud_name]
    return tr_pkl_paths, val_pkl_paths
