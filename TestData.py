import os
import torch
from torch.utils import data
import librosa
import numpy as np

class SingleMus(data.Dataset):
    def __init__(self, folderPath):
        
        mus_list = [i for i in os.listdir(folderPath) if i.endswith('.wav')][1000:1500]
        self.data = []
        for mus in mus_list:
            sub = self.load_trim(folderPath + mus)
            if isinstance(sub, np.ndarray):
                for row in sub:
                    self.data.append(row)
        
        print('Data size:', len(self.data) )
        
        
    def load_trim(self, wavpath, size = 20):
        try:
            c, cr = librosa.load(wavpath)
        except :
            print('errorfile:',wavpath)
            return 
        trim_len = int(np.floor(len(c)/512)*512)
        c = c[:trim_len].reshape(-1, 512)
        sections = np.random.randint(len(c), size = size)
        sub = c[sections]   
        
        return sub
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]
