import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
from models.resnet18_keras import SelfAttention


model_dir = os.path.join('saved-models', 'voxceleb1_attention_vgg_dropout_keras_fullset.h5')
data_dir = '/media/data/bbekci/voxceleb/pkls_colwise_normed/'
embed_main_dir = '/media/data/bbekci/voxceleb_id_embeds_vgg/'


def load_audio_pickle(ppath):
    
    with open(ppath, 'rb') as pickle_load:
        loaded_sample = pickle.load(pickle_load)
    
    idname, videoname, features = loaded_sample

    feature_len = features.shape[0]
    iter_count = feature_len // 300
    
    sample_features = np.zeros((iter_count, 300, 40, 1))
    for i in range(iter_count):
        feature_start = i * 300
        feature_end = (i+1) * 300
        sample_features[i] = np.expand_dims(features[feature_start:feature_end], axis=-1)

    return sample_features

model = tf.keras.models.load_model(model_dir, custom_objects={'SelfAttention': SelfAttention, 'GlorotUniform': tf.keras.initializers.GlorotUniform()})
model.summary()

saved_model = Model(model.input, model.get_layer('dense').output)

pids = os.listdir(data_dir)

for pid in pids:
    pid_path = os.path.join(data_dir, pid)
    p_embed = np.zeros((1, 256))
    total_audios = 0
    video_names = os.listdir(pid_path)
    for video_name in video_names:
        video_path = os.path.join(pid_path, video_name)
        audio_names = os.listdir(video_path)
        for audio_name in audio_names:
            total_audios += 1
            audio_path = os.path.join(video_path, audio_name)
            # load wav file first
            loaded_wav = load_audio_pickle(audio_path)
            preds = saved_model.predict(loaded_wav)

            #mean_embeds = np.mean(preds, axis=0)
            p_embed += np.sum(preds, axis=0)

    dest_path = os.path.join(embed_main_dir, pid)
    os.makedirs(dest_path, exist_ok=True)
    dest_pickle_path = os.path.join(dest_path, audio_name)
    mean_embeds = p_embed / total_audios
    with open(dest_pickle_path, 'wb') as pickle_file:
        pickle.dump([mean_embeds, pid], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
