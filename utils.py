import os
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal


"""

a frame-length of 25 ms and a frame-shift of 10 ms are
extracted. Since the utterances in the VoxCeleb dataset are
of varying duration (up to 144.92 s), we x the length of
the input sequence to 3 seconds. These end up as log-mel
lterbank of size 40300 for a 3-second utterance.
frame-shift = hop time ???      
n_fft=length of the FFT window(?=frame length)(# Size of the FFT may be used as the window length)
win_length= The window will be of length win_length


hop_length = number of samples between successive frames
( Step or stride between windows. If the step is smaller than the window lenght, the windows will overlap)
https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

"""

def apply_melspectrogram_to_file(filename):
    y, sample_rate = librosa.load(filename,duration=3)
    duration=len(y) / sample_rate
    print(duration)

    librosa.display.waveplot(y=y,sr=sample_rate)
    if y.shape[0] == 0:
        print("y.shape[0] == 0")
        return None
    else:
        print(y.shape)
        window_time = .025
        hop_time = .01
        n_fft = sample_rate * window_time
        hop_len = sample_rate*hop_time
        # print(int(n_fft))
        
        melspectrogram = librosa.feature.melspectrogram(y=librosa.effects.preemphasis(y), sr=sample_rate, n_mels=40,
         n_fft=int(n_fft), hop_length = int(hop_len),window=signal.windows.hamming)
        #melspectrogram = librosa.feature.melspectrogram(y=librosa.effects.preemphasis(y), sr=sample_rate)
        log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        normalized_melspectrogram = (log_melspectrogram - log_melspectrogram.mean()) / log_melspectrogram.std()



    melspectrogram=normalized_melspectrogram

    
    
    return melspectrogram

def display_spectrogram(spectrogram):
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='s')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')