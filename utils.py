import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa
import matplotlib.pyplot as plt




def apply_melspectrogram_to_file(filename):
    y, sr = librosa.load(filename)
    if y.shape[0] == 0:
        return None
    else:
        # print(y.shape)
        window_time = .025
        hop_time = .01
        n_fft = sr * window_time
        hop_len = sr*hop_time
        # print(int(n_fft))
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=int(n_fft), hop_length = int(hop_len))
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    
    
    return spectrogram

def display_spectrogram(spectrogram):
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')