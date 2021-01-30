import os
import torch
from torch.utils import data
import librosa
import numpy as np
from IPython.display import clear_output

class Data_noiseTIMIT_random(data.Dataset):
    def __init__(self, task, overlap = 64, level=0, window = False):
        
        self.task = task
        # Set small datasets
        if task == 'train':
            folderPath = '/media/sdc1/Data/timit-wav/train'
            stop = 300
        elif task == 'test':
            folderPath = '/media/sdc1/Data/timit-wav/test'
            stop = 30
        self.max_x = 0
        
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
        

        names = ['birds', 'computerkeyboard', 'jungle', 'ocean', 'casino', 'eatingchips', 'machineguns',\
                 'cicadas', 'frogs', 'motorcycles']

        self.noise = []
        for idx in range(len(names)):
            noise_path = '/media/sdc1/Data/Duan/{}.wav'.format(names[idx])
            n, nr = librosa.load(noise_path, sr=None)
            self.noise.append(n)

        self.data_c = []
        self.data_x = []
        self.data_c_l = []
        self.data_x_l = []
        
        # Extract and Trim section and 
        for i, uttr in enumerate(uttr_list):
            if i == stop:
                break     
            clear_output(wait=False)
            os.system('clear')
            print('Preparing data files:' + str(i)+'/300', flush=True)
            self.load_trim(uttr, level=level)
        
        self.data_c_l = np.array(self.data_c_l)
        self.data_x_l = np.array(self.data_x_l)
#         self.data_c /= self.max_x
#         self.data_x /= self.max_x
        self.data_c_l /= self.max_x
        self.data_x_l /= self.max_x

        print('Data size:', len(self.data_c_l))
        
    
    def load_trim(self, wavpath, size = 1, level = 0):
        
        try:
            c, cr = librosa.load(wavpath, sr = None)
            c /= np.std(c) # Normalizing c std=1
        except OSError:
            print('errorfile:',wavpath)
            return
        
        scalar = 10.0 ** (0.05 * (level))
        
        for n in self.noise:
            n = n[len(n)//2:len(n)//2+len(c)]
            n /= np.std(n)* scalar
            self.array_append(c, n)
        
    def array_append(self, c, n):
    
        x = c + n
        self.max_x = max(self.max_x, abs(max(x)))
        
        c_l = []
        x_l = []
        
        for i in range(0, len(c), 512 - self.overlap):
            if i + 512 > len(c):
                break
            c_l.append(c[i:i+512])
            x_l.append(x[i:i+512])

#         c = c[:len(c_l)*(512-self.overlap)+self.overlap]
#         x = x[:len(c)]
        
        if self.windowOn:
            c_l = c_l * self.window
            x_l = x_l * self.window
        
#         self.data_c.append(c)
#         self.data_x.append(x)
        self.data_c_l.extend(c_l)
        self.data_x_l.extend(x_l)
        
        
    def __len__(self):
        if self.task == 'train':
            return len(self.data_c_l)
        if self.task == 'test':
            return len(self.data_c_l)

    def __getitem__(self, idx):

        return self.data_c, self.data_x, self.data_c_l[idx], self.data_x_l[idx]
