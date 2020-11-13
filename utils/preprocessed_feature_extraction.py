import os
import glob
import torch
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from torch.utils.data import random_split, Dataset, DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def display_spectrogram(spectrogram):
    librosa.display.specshow(spectrogram.transpose(), hop_length=220.5,y_axis='mel', fmax=8000, x_axis='s')
    #getting 7 second in time axis, it should be 3, why???
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def logmel_filterbanks(filename,pre_emphasis=0.97,frame_size = 0.025,frame_stride = 0.01,nfilt=40,normalize=True):
    target_len = 66150
    
    signal, sample_rate = librosa.load(filename)

    while(signal.shape[0] < target_len):
        signal = np.append(signal, signal[:target_len - signal.shape[0]])
    
    #Pre-Emphasis step
    emphasized_signal = np.empty(shape=len(signal)+1)
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    #Framing
    
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    #Hamming-Window
    frames *= np.hamming(frame_length)
    
    #FFT
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    
    #Filter-Bank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    if normalize==True:
        #filter_banks = (filter_banks - filter_banks.mean()) / (filter_banks.max() - filter_banks.min())
        normed_filter_banks = (filter_banks - filter_banks.mean(axis=0)) / filter_banks.std(axis=0)
        return normed_filter_banks

    
    return filter_banks

def mfcc(filter_banks,num_ceps=13):
    return dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]

if __name__=='__main__':
    
    dataset_dir = '/home/bbekci/datasets/vctk/wav48_silence_trimmed'
    data = []
    c2i, i2c = {}, {}
    for indx, cla in enumerate(os.listdir(dataset_dir)):
                main_path = dataset_dir + '/' + cla + '/*.flac'
                for file_path in glob.glob(main_path):
                    data.append((file_path, cla))
                c2i[cla] = indx
                i2c[indx] = cla


    with open('preprocessed_vctk.pkl', 'wb') as pickle_file:
        result=[]
        for i in range(0,len(data)):
                sample = []
                sound_path, class_name = data[i]
                sound_data = logmel_filterbanks(sound_path)
                label = c2i[class_name]

                sample = [label, sound_data]

                result.append((sample))

        pickle.dump(result, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    class PreprocessedDataset(Dataset):
        def __init__(self, file_dir):
            self.file_dir = file_dir
            self.lst = 0
            with open(file_dir, 'rb') as pickle_load:
                self.lst = pickle.load(pickle_load)

        def __len__(self):
            return len(self.lst)

        def n_class(self):
            return self.lst[-1][0]

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            sound_data = self.lst[idx][1]
            label = self.lst[idx][0]

            sample = (sound_data, label)

            return sample

    dataset_dir = '/home/bbekci/inzpeech/preprocessed_vctk.pkl'
    offset_dict = {}
    max_epochs = 25
    batch_size = 256

    sound_data = PreprocessedDataset(file_dir=dataset_dir)

    n_classes = sound_data.n_class()


    train_data, test_data = random_split(sound_data,
                                         [int(len(sound_data) * 0.8),
                                          len(sound_data) - int(len(sound_data) * 0.8)]
                                         )

    train_dataset_loader = torch.utils.data.DataLoader(train_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=4)

    test_dataset_loader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=4)


