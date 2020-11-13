import os
import pickle
from preprocessed_feature_extraction import logmel_filterbanks

dataset_dir = "/media/data/bbekci/voxceleb/wav/"
output_dir = "/media/data/bbekci/voxceleb/pkls_colwise_normed"

def process_id(idpath, idname):
    video_names = os.listdir(idpath)
    for video_name in video_names:
        video_path = os.path.join(idpath, video_name)
        
        audio_names = [fname for fname in os.listdir(video_path) if fname.endswith('.wav')]
        audio_paths = [os.path.join(video_path, audio_name) for audio_name in audio_names if audio_name.endswith('.wav')]
        
        for i, audio_path in enumerate(audio_paths):

            features =  logmel_filterbanks(audio_path)
            dest_path = os.path.join(output_dir, idname, video_name)
            os.makedirs(dest_path, exist_ok=True)
            full_dest_path = os.path.join(dest_path, audio_names[i].replace('.wav','.pkl'))
            with open(full_dest_path, 'wb') as pickle_file:
                pickle.dump([idname, video_name, features], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        
os.makedirs(output_dir, exist_ok=True)
ids = [fname for fname in os.listdir(dataset_dir) if 'id' in fname]
id_paths = [os.path.join(dataset_dir, nid) for nid in ids]

for i, idp in enumerate(id_paths):
    print("Process person: ", ids[i])
    process_id(idp, ids[i])
    
