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
win_length= The window will be of length n_fft if not specified


hop_length = number of samples between successive frames
( Step or stride between windows. If the step is smaller than the window lenght, the windows will overlap)
https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html

"""

def return_files(directory):

    all_files=[]
    for file in os.listdir(os.getcwd()+'/'+directory):
        if file.endswith(".wav"):
            all_files.append(os.getcwd()+'/'+directory+'/'+file)
    print(all_files)
    return all_files

def return_spectograms(directory):
    files=return_files(directory)
    spectograms=[]
    for file in files:
        spectograms.append(apply_melspectrogram_to_file(file))
    return spectograms


    
    

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
        print(n_fft)
        hop_len = sample_rate*hop_time
        #print(int(n_fft))
        
        melspectrogram = librosa.feature.melspectrogram(y=librosa.effects.preemphasis(y), sr=sample_rate, n_mels=40,n_fft=int(n_fft), hop_length = int(hop_len),window=signal.windows.hamming)
        #melspectrogram = librosa.feature.melspectrogram(y=librosa.effects.preemphasis(y), sr=sample_rate,n_mels=40,window=signal.windows.hamming)
        log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        #normalized_melspectrogram = (log_melspectrogram - log_melspectrogram.mean()) / log_melspectrogram.std()



    melspectrogram=log_melspectrogram.transpose()[:-1]
    print(melspectrogram.shape)

    
    
    return melspectrogram
def display_all_spectograms(spectrograms):
    for i in range(0,len(spectrograms)):
        display_spectrogram(spectrograms[i])

def display_spectrogram(spectrogram):
    librosa.display.specshow(spectrogram.transpose(), y_axis='mel', fmax=8000, x_axis='s')
    #getting 7 second in time axis, it should be 3, why???
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()