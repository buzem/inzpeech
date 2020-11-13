# Speaker Identificition
inzva AI Projects #5 - Speaker Identification

## Project Description
In this project we tried to solve the problem of Speaker Identification which is a process of recognizing a person from a voice utterance. We implemented the methods propsed in [Deep CNNs With Self-Attention for Speaker Identification](https://ieeexplore.ieee.org/document/8721628) paper on both Tensorflow-Keras and Pytorch.


## Dataset

We used below datasets: 

* [VCTK Corpus](https://datashare.is.ed.ac.uk/handle/10283/3443)
* [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/)


VCTK dataset is easy to use, no license agreement is required and it is easy to use after download. 

For the VoxCeleb dataset, it is recommended to visit its website to sign up and find download and conversion scripts for the datasets. 

The data [split text file](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt) for identification will be required. 

The files under [dataloaders](dataloaders/) used for loading the data with datagens in Keras and dataloaders in Pytorch. The scripts can generate file paths in runtime or read from a txt file directly. It is recommended to generate txt files. Check [this notebook](Save_VoxCelebTxts.ipynb) to generate such a file.

It is also recommended to generate pickle files from audio features first and load them. Our data loaders works with that way too. Check out scripts under [utils](utils/) folder to create such files.


## Preprocess

Before feeding the audio files into our models, we extract filter bank coefficients from them. Check out [here](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html) for the complete process. Our implementation is under [utils/preprocessed_feature_extraction.py](utils/preprocessed_feature_extraction.py)

## Models
We implemented below architectures:

* [VGG-like CNN](models/model_keras.py)
* [ResNet18](models/resnet18_keras.py)
* [ResNet50](ResNet/model.py)


## Results

We achieved 

## Nearest Neighbor Search

After training our models, we extracted embeddings with the trained model and used knn algorithm to find closest neighboors of the extracted embeddings. Such system can be used to find the closest voice utterances and their class labels for a given audio signal. 

Check out [extract_embeds.py](extract_embeds.py) and [closest_celeb.py](closest_celeb.py) scripts for the implementation of this method.


## Project Dependencies

 - Keras
 - Pytorch
 - MatPlotLib
 - TensorFlow
 - Pickle
 - Numpy
 - Librosa