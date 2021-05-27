import os
import torch
from torch.utils import data
import librosa
import numpy as np
from IPython.display import clear_output

class Data_TIMIT(data.Dataset):
    def __init__(self, task, mix_num = 1, overlap = 64, level=0, window = False):
        
        self.task = task
        # Set small datasets
        if task == 'train':
            folderPath = '/media/sdc1/Data/timit-wav/train'
            stop = 1000
        elif task == 'test':
            folderPath = '/media/sdc1/Data/timit-wav/test'
            stop = 300
        self.max_x = 0
        self.overlap = overlap
        
        # Extract uttr files 
        
        SpkrFolders = []
        for dr in range(8):
            rootPath = folderPath+'/dr{}'.format(dr+1)
            SpkrFolders += [os.path.join(rootPath,spkr) for spkr in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, spkr))]
        
        
        self.uttr_list = []
        for folderPath in SpkrFolders:
            self.uttr_list += ['/{}/{}'.format(folderPath, i) for i in os.listdir(folderPath) if i.endswith('.wav')]
        
        # Noise
#         idx = 4
        names = ['birds', 'computerkeyboard', 'jungle', 'ocean', 'casino', 'eatingchips', 'machineguns',\
                 'cicadas', 'frogs', 'motorcycles']

        self.noise = []
        for idx in range(len(names)):
            noise_path = '/media/sdc1/Data/Duan/{}.wav'.format(names[idx])
            n, nr = librosa.load(noise_path, sr=None)
            self.noise.append(n)
        
    def __len__(self):
        if self.task == 'train':
            return 1000
        if self.task == 'test':
            return 300

    def __getitem__(self, idx):
        
        while 1:
            try:
                uttr = self.uttr_list[idx]
                c, cr = librosa.load(uttr, sr = None)
                c /= np.std(c) # Normalizing c std=1
                break
            except OSError:
                print('errorfile:',wavpath)
                idx += 1
        level = 0
        scalar = 10.0 ** (0.05 * (level))
        
        n_id = np.random.randint(10, size=1)[0]
#         print(n_id)
        n = self.noise[n_id]
        n = n[len(n)//2:len(n)//2+len(c)]
        n /= np.std(n)* scalar
        
        x = c + n
        c_l = []
        x_l = []
        
        for i in range(0, len(c), 512 - self.overlap):
            if i + 512 > len(c):
                break
            c_l.append(c[i:i+512])
            x_l.append(x[i:i+512])
        c_l = np.array(c_l)
        x_l = np.array(x_l)
        
        length = (len(c_l)-1)*(512-self.overlap)+512
        c = c[:length]
        x = x[:length]
        
        return c, x, c_l, x_l