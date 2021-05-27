import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.cm as cm
import pandas as pd
import os
import librosa
import librosa.display
import torch.nn.functional as F
from torch.autograd import Variable
import time
import IPython.display as ipd
from IPython.display import clear_output
import math
import soundfile as sf


def SISDR(s,sr,  cuda = False):
    
    eps = 1e-20
    scale = torch.sum(sr * s, dim = 1) / torch.sum(s**2, dim = 1) 
    scale = scale.unsqueeze(dim = 1) # shape - [50,1]
    s = s * scale
    sisdr = torch.mean(10*torch.log10(torch.sum(s**2, dim = 1)/(torch.sum((s-sr)**2, dim=1)+eps)+eps))
    if cuda:
        return torch.mean(sisdr)
    else:
        return torch.mean(sisdr).cpu().data.numpy()

def SDR(s, sr, cuda = False): # input (50, 512), (50, 512)
    
    eps=1e-20
    sdr = torch.mean(10*torch.log10(torch.sum(s**2, dim = 1)/(torch.sum((s-sr)**2, dim=1)+eps)+eps))
    
    if cuda:
        return sdr
    else:
        return sdr.cpu().data.numpy()


def l1Loss(s, sr): # input waveform(torch.Tensor)
    
    length = min(len(s),len(sr))
    s = s[:length]
    sr = sr[:length]
    
    loss = torch.sum(torch.abs(s-sr))/length
    

    return loss


def rebuild(output, overlap = 32):
    output = output.cpu()
    len_wav = len(output) * (512 - overlap) + overlap
    wave = torch.zeros(len_wav)
    for i in range(len(output)):
#         print(i)
        wave[i*(512-overlap):i*(512-overlap)+512] += output[i]       
    
    return wave

def rebuild_extraWindow(output, overlap = 32):
    
    output = output.detach().cpu().data.numpy()
    len_wav = len(output) * (512 - overlap) + overlap
    
    window = np.hanning(overlap*2-1) 
    window = np.concatenate((window[:overlap],np.ones(512-overlap*2),window[overlap-1:]))
    window = window.reshape(1,-1)
    window = window.astype(np.float32)
    output *= window
    
    wave = np.zeros(len_wav)
    for i in range(len(output)):
#         print(i)
        wave[i*(512-overlap):i*(512-overlap)+512] += output[i]
    
    return torch.tensor(wave, dtype=torch.float).requires_grad_()


import collections

def cal_entropy(arg):
    
    entropy = 0
    counter = collections.Counter(arg)
    sum_v = sum(counter.values())
    for value in counter.values():
        p = value / sum_v
        entropy += - np.log2(p)*p
        
    return entropy


def entropy_prob(prob): # prob (bs, 512, num_m)
    
    entropy = 0
    eps = 1e-20
    prob_counter = torch.sum(prob, dim=0)
    prob_counter = torch.sum(prob_counter, dim=0)
    assert len(prob_counter) == prob.shape[-1]
    prob_counter = prob_counter / sum(prob_counter)
    entropy = - torch.sum(torch.log2(prob_counter+eps)*prob_counter)
#     sum_v = sum(prob_counter)
#     for value in prob_counter:
#         p = value / sum_v
#         entropy += -torch.log(p+eps)*p
    
    return entropy

def mulaw_loss(s, sr, f_SDR):
    import math
    mu = 1024 # 
    s = s * (torch.log(1 + mu*torch.abs(s))/(math.log(1+mu))) # s or tanh(s)
    sr = sr * (torch.log(1 + mu*torch.abs(sr))/(math.log(1+mu))) # s or tanh(s)
    
    return f_SDR(s, sr)

def pesq(reference, degraded, sample_rate=None, program='pesq'):
    """ Return PESQ quality estimation (two values: PESQ MOS and MOS LQO) based
    on reference and degraded speech samples comparison.
    Sample rate must be 8000 or 16000 (or can be defined reading reference file
    header).
    PESQ utility must be installed.
    """
    if not os.path.isfile(reference) or not os.path.isfile(degraded):
        raise ValueError('reference or degraded file does not exist')
    if not sample_rate:
        import wave
        w = wave.open(reference, 'r')
        sample_rate = w.getframerate()
        w.close()
    if sample_rate not in (8000, 16000):
        raise ValueError('sample rate must be 8000 or 16000')
    import subprocess
    args = [program, reference, degraded, '+%d' % sample_rate, '+wb']
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    out = out.decode("utf-8")
    last_line = out.split('\n')[-2]
    pesq_wb = float(last_line.split()[-1:][0])
    return pesq_wb
  

def load_trim(wavpath, n, db=0, overlap=32, window_in_data = False):
    
    window = np.hanning(overlap*2-1) 
    window = np.concatenate((window[:overlap],np.ones(512-overlap*2),window[overlap-1:]))
    window = window.reshape(1,-1)
    window = window.astype(np.float32)
    
    max_x = 48
    
    try:
        c, cr = librosa.load(wavpath, sr = None)
        c /= np.std(c)
    except OSError:
        print('errorfile:',wavpath)
    
    n = n[len(n)//2:len(n)//2+len(c)]        
        
    scalar = 10.0 ** (0.05 * (db))
    n /= np.std(n)* scalar
        
#     n /= max(abs(c))
#     c /= max(abs(c))
    x = c + n
    
    c_l = []
    x_l = []
    
    for i in range(0, len(c), 512 - overlap):
        if i + 512 > len(c):
            break
        c_l.append(c[i:i+512])
        x_l.append(x[i:i+512])
    c_l = np.array(c_l)
    x_l = np.array(x_l)
    c = c[:len(c_l)*(512-overlap)+overlap]
    x = x[:len(c)]
    
    if window_in_data:
        c_l = c_l * window
        x_l = x_l * window

    return c/max_x, x/max_x, c_l/max_x, x_l/max_x

def gen_sound(i=0,db=0, window_in_data=False, overlap=32):
    wavpath0 = '/media/sdc1/Data/timit-wav/test/dr5/mrws1/sx140.wav'
    wavpath1 = '/media/sdc1/Data/timit-wav/test/dr1/faks0/sa1.wav'
    wavpath2 = '/media/sdc1/Data/timit-wav/test/dr1/mreb0/sa2.wav'
    wavpath3 = '/media/sdc1/Data/timit-wav/test/dr3/mkch0/sx28.wav'
    wavpath4 = '/media/sdc1/Data/timit-wav/test/dr3/fkms0/sx50.wav'
    wavpath5 = '/media/sdc1/Data/timit-wav/test/dr5/fjcs0/sx139.wav'
    wavpath6 = '/media/sdc1/Data/timit-wav/test/dr4/fsem0/sx28.wav'
    wavpath7 = '/media/sdc1/Data/timit-wav/test/dr4/mkcl0/sx191.wav'
    wavpath8 = '/media/sdc1/Data/timit-wav/test/dr6/flnh0/sx134.wav'
    wavpath9 = '/media/sdc1/Data/timit-wav/test/dr6/mesd0/sx12.wav'

    path_name = [wavpath0, wavpath1, wavpath2,wavpath3,wavpath4,wavpath5,wavpath6,wavpath7,wavpath8,wavpath9]
    # n_idx = [2, -1, -3, -4, 5, 1]

    # idx = 4
    names = ['birds', 'computerkeyboard', 'jungle', 'ocean', 'casino', 'eatingchips', 'machineguns',\
                     'cicadas', 'frogs', 'motorcycles']
#     i = 1
    path = path_name[i]
    noise_path = '/media/sdc1/Data/Duan/{}.wav'.format(names[i])
    n, nr = librosa.load(noise_path, sr=None)
    # tes, sr = librosa.load(wavpath5, sr=None)

#     db = 5
    c, x, c_l, x_l = load_trim(path, n, db=db, overlap=32, window_in_data=window_in_data)

    c_l = torch.Tensor(c_l)
#     c = torch.Tensor(c)
    x_l = torch.Tensor(x_l)
    #         x = torch.Tensor(x)
    
    return c, x, c_l, x_l


def generate_result(model, x_l, window_in_data, overlap, soft):
    
    model.eval()
    n_h = None
#     s_h, n_h, prob_s, prob_n = model(x_l.cuda(),soft = False)
    with torch.no_grad():
        output = model(x_l.cuda(),soft = soft)
    if window_in_data:
        rebuild_f = rebuild
    else:
        rebuild_f = rebuild_extraWindow
        
    s_h = output[0]
    s_h = s_h[:,0,:]
    if len(output) == 4:
        n_h = output[1]
        n_h = n_h[:,0,:]
    rs = rebuild_f(s_h, overlap=overlap).cpu().data.numpy()
    if not isinstance(n_h, type(None)):
        rx = rebuild_f(s_h+n_h, overlap=overlap).cpu().data.numpy()
    else:
        rx = rs
    
    return rs, rx

def calculate_pesq(model, window_in_data, soft, model_name):
    
    pesq_list = []

    with torch.no_grad():
        model.eval()
        for i in range(10):
            data = gen_sound(i, window_in_data, overlap=32) # c, x, c_l, x_l / c, c_l
            inp = data[-1] if 'Prop' in model_name else data[-2]
            rs, rx = generate_result(model, inp, window_in_data, 32, soft)
            sf.write('../training_samples/ori_sig_'+model_name+'.wav', data[0], 16000, 'PCM_16')
            sf.write('../training_samples/dec_sig_'+model_name+'.wav', rx, 16000, 'PCM_16')
            the_pesq = pesq('../training_samples/ori_sig_'+model_name+'.wav','../training_samples/dec_sig_'+model_name+'.wav', 16000)
            pesq_list.append(the_pesq)
    
    return np.mean(pesq_list)

def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)

def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)

# generate Mel filter bank
def melFilterBank(numCoeffs, fftSize = None):
    
    SAMPLE_RATE = 16000
    minHz = 0
    maxHz = SAMPLE_RATE / 2            # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = WINDOW_SIZE
    else:
        numFFTBins = fftSize // 2 + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.array(range(numCoeffs + 2))
    melRange = melRange.astype(np.float32)
    
#     print(melRange)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel
#     print(melCenterFilters)

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = int(math.floor(numFFTBins * melCenterFilters[i] / maxHz))
#         print(melCenterFilters[i])x
    
    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, int(numFFTBins)))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)
        
        startRange = int(melCenterFilters[i - 1])
        midRange   = int(melCenterFilters[i])
        endRange   = int(melCenterFilters[i + 1])
        
        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))
        
        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat

def melMSELoss(s, sr, n_mels = [8, 16, 32, 128]): 
    
    assert s.shape[1] == 512
    assert sr.shape[1] == 512
    
    loss = 0
    eps = 1e-20
    mse = nn.MSELoss().cuda()
    s = s.cuda()
    sr = sr.cuda()
    
    def no_window(length):
        return torch.ones(length)
    
    FFT_SIZE = 512
    
    MEL_FILTERBANKS = []
    for scale in n_mels:
        filterbank_npy = melFilterBank(scale, FFT_SIZE).transpose()
        filterbank_npy = torch.tensor(filterbank_npy, dtype=torch.float, device='cuda')
        MEL_FILTERBANKS.append(filterbank_npy)

    spec = torchaudio.transforms.Spectrogram(n_fft=512, window_fn=no_window).cuda()
    
    s_spec = spec(s)[:,:,1]/FFT_SIZE
    sr_spec = spec(sr)[:,:,1]/FFT_SIZE

    for filterbank in MEL_FILTERBANKS:
        # s_spec.shape -> [Batch, 257]
        # filterbank -> [257, 8]
        
        s_melspec = torch.log(torch.matmul(s_spec, filterbank) + 1e-20)
        sr_melspec = torch.log(torch.matmul(sr_spec, filterbank) + 1e-20)

        error = torch.sqrt(mse(s_melspec, sr_melspec)+1e-20)
        loss += error
        
    return loss/len(n_mels)


def melMSELoss_short(s, sr, n_mels = [8, 16, 32, 128]): 

    assert s.shape[1] == 512
    assert sr.shape[1] == 512

    def no_window(length):
        return torch.ones(length)

    loss = 0
    eps = 1e-20
    mse = nn.MSELoss().cuda()
    s = s.cuda()
    sr = sr.cuda()
    for n in n_mels:
        melspec = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=n, window_fn=no_window).cuda() 
        s_mel = melspec(s)[:,:,1] # shape - [bt, n] 
        sr_mel = melspec(sr)[:,:,1]
        # s_mel.shape -> [bt, n]

        s_mel = torch.log(s_mel + eps)
        sr_mel = torch.log(sr_mel + eps)
        error = mse(s_mel, sr_mel) #/len(s_mel)
        loss += error

    return loss/len(n_mels)