import os
import fnmatch
import pickle5 as pickle
import numpy as np
from scipy.spatial.distance import cdist

def load_embeds(embeding_main_path):
    pkl_matches = []
    for root, dirname, filenames in os.walk(embeding_main_path):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            pkl_matches.append(os.path.join(root, filename))
    return pkl_matches

class NearestNeighboor(object):

    @classmethod
    def init_neighbor(cls, data_path):
        pkl_paths = load_embeds(data_path)
        cls.labels = []
        #self.embeds = np.zeros((len(pkl_paths), 256))
        cls.embeds = np.zeros((len(pkl_paths), 256))

        for i, p in enumerate(pkl_paths):
            print("Progress: ", i , " / ", len(pkl_paths), end='\r')
            with open(p, 'rb') as pfile:
                loaded_pkl = pickle.load(pfile)
                cls.labels.append(loaded_pkl[1])
                cls.embeds[i] = loaded_pkl[0]

    @classmethod
    def closest_labels(cls, test_sample, k):
        # Get euclidean distances as 2D array
        dist = cdist(cls.embeds, test_sample, 'sqeuclidean').reshape(-1)
        # Find the k smallest distances
        indx = np.argpartition(dist, k)[: k]
        return np.unique(np.array(cls.labels)[indx])