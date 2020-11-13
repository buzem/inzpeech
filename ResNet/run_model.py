import os
import glob
import torch
import librosa

import numpy as np
import pandas as pd
import scipy.signal as signal

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from model import Net_ResNet50

from torch.utils.data import random_split, Dataset, DataLoader
from tqdm import tqdm

# Parameters
dataset_dir = '/home/bbekci/datasets/vctk/wav48_silence_trimmed'
max_epochs = 100
batch_size = 64

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True


class VCTKData(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.c2i, self.i2c = {}, {}
        for indx, cla in enumerate(os.listdir(root_dir)):
            main_path = root_dir + '/' + cla + '/*.flac'
            for file_path in glob.glob(main_path):
                self.data.append((file_path, cla))

            self.c2i[cla] = indx
            self.i2c[indx] = cla

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def n_class(self):
        return len(list(self.c2i.keys()))

    # According to our input 66150 is the length
    def apply_melspectrogram(self, filename):
        target_len = 66150
        y, sample_rate = librosa.load(filename, duration=3)
        
        while(y.shape[0] != target_len):
            y = np.append(y, y[:target_len - y.shape[0]])

        if y.shape[0] == 0:
            print("y.shape[0] == 0")
            return None

        window_time = .025
        hop_time = .01
        n_fft = int(sample_rate * window_time)

        hop_len = int(sample_rate * hop_time)

        melspectrogram = librosa.feature.melspectrogram(y=librosa.effects.preemphasis(y),
                                                        sr=sample_rate,
                                                        n_mels=40,
                                                        n_fft=n_fft,
                                                        hop_length=hop_len,
                                                        window=signal.windows.hamming)
        log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

        melspectrogram = log_melspectrogram.T[:-1]

        out = np.expand_dims(melspectrogram, axis=0)

        return out

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sound_path, label = self.data[idx]
        sample = (self.apply_melspectrogram(sound_path), self.c2i[label])

        if self.transform:
            sample = self.transform(sample)

        return sample


sound_data = VCTKData(root_dir=dataset_dir)
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


net = Net_ResNet50(img_channel=1, num_classes=n_classes)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters())

for epoch in range(max_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataset_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
