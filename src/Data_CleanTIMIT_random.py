import os
import torch
from torch.utils import data
import librosa
import numpy as np
from IPython.display import clear_output

class Data_CleanTIMIT_random(data.Dataset):
    def __init__(self, task, mix_num = 1, overlap = 64, level=0, window = False):
        
        self.task = task
        # Set small datasets
        if task == 'train':
            folderPath = '/media/sdc1/Data/timit-wav/train'
            stop = 3000
        elif task == 'test':
            folderPath = '/media/sdc1/Data/timit-wav/test'
            stop = 500
        self.max_c = 0
        
        self.windowOn = window
        
        self.overlap = overlap
        window = np.hanning(self.overlap*2) 
        self.window = np.concatenate((window[:overlap],np.ones(512-overlap*2),window[overlap-1:]))
        self.window = self.window.reshape(1,-1)
        self.window = self.window.astype(np.float32)
        
        # Extract uttr files 
        
        SpkrFolders = []
        for dr in range(8):
            rootPath = folderPath+'/dr{}'.format(dr+1)
            SpkrFolders += [os.path.join(rootPath,spkr) for spkr in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, spkr))]
        
        
        uttr_list = []
        for folderPath in SpkrFolders:
            uttr_list += ['/{}/{}'.format(folderPath, i) for i in os.listdir(folderPath) if i.endswith('.wav')]
        
        self.data_c = []
        self.data_c_l = []
        
        # Extract and Trim section and 
        for i, uttr in enumerate(uttr_list):
            if i == stop:
                break     
            clear_output(wait=False)
            print(i, flush=True)
            self.load_trim(uttr, level=level)
        
#         self.data_c /= self.max_c
        self.data_c_l /= self.max_c
        
        print('Data size:', len(self.data_c))
        
    
    def load_trim(self, wavpath, size = 1, level = 0):
        try:
            c, cr = librosa.load(wavpath, sr = None)
            c /= np.std(c) # Normalizing c std=1
        except OSError:
            print('errorfile:',wavpath)
            return
        
        self.max_c = max(self.max_c, abs(max(c)))

        c_l = []
        
        for i in range(0, len(c), 512 - self.overlap):
            if i + 512 > len(c):
                break
            c_l.append(c[i:i+512])
        c_l = np.array(c_l)
        c = c[:len(c_l)*(512-self.overlap)+self.overlap]
        
        if self.windowOn:
            c_l = c_l * self.window
        
        self.data_c.append(c)
        self.data_c_l.append(c_l)        
        
    def __len__(self):
        if self.task == 'train':
            return len(self.data_c_l)
        if self.task == 'test':
            return len(self.data_c)

    def __getitem__(self, idx):
        return self.data_c[idx], self.data_c_l[idx]
    