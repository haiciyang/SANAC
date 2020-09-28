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

def melMSELoss(s, sr): # input waveform(torch.Tensor)
    
    n_mels = [8, 16, 32, 128]
    loss = 0
    eps = 1e-20
    mse = nn.MSELoss()
    for n in n_mels:
        s_mel = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=n)(s)
        sr_mel = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=n)(sr)
        s_mel = torch.log(s_mel + eps)
        sr_mel = torch.log(sr_mel + eps)
        m = mse(s_mel, sr_mel)
        loss += m
    return loss/len(n_mels)    

def rebuild(output, overlap = 64):
    output = output.cpu()
    len_wav = len(output) * (512 - overlap) + overlap
    wave = torch.zeros(len_wav)
    for i in range(len(output)):
#         print(i)
        wave[i*(512-overlap):i*(512-overlap)+512] += output[i]       
    
    return wave

def test_newData(soft, test_loader, model):
#     model.eval()
    sdr_list_s = []
#     sdr_list_n = []
    sdr_list_rs = []
    sdr_list_rx = []
    sdr_list_diff = []
    max_rs_sdr = 0
    max_rx_sdr = 0
    max_rs = None
    max_rx = None

    i = 0
    for wave_s, wave_x, source, mixture in test_loader: 

        source = source[0]
        mixture = mixture[0]
#         print(data)
        noise = (mixture-source).cuda()
        source = source.cuda()
        mixture = mixture.cuda()
        s_h, n_h, arg_s, arg_n = model(mixture, soft=soft)
        
        sdr_list_s.append(SISDR(source, s_h))    
#         sdr_list_n.append(SISDR(noise, n_h))
        
        rs = rebuild(s_h).unsqueeze(dim = 0)
        rn = rebuild(n_h).unsqueeze(dim = 0)
        
        rs_sdr = SISDR(wave_s, rs)
        rx_sdr = SISDR(wave_x, rs+rn)
        sdr_list_rs.append(rs_sdr)
        sdr_list_rx.append(rx_sdr)
        
        start_sdr = SISDR(wave_s, wave_x/max(wave_x[0]))
        sdr_list_diff.append(rs_sdr-start_sdr)
        
        if max_rs_sdr < rs_sdr:
            max_rs_sdr = rs_sdr
            max_rs = (wave_x, rs, rn)
#         if max_rx_sdr < rx_sdr:
#             max_rx_sdr = rx_sdr
#             max_rx = (rs, rn)
         
    
    s_score = np.mean(sdr_list_s) 
#     n_score = np.mean(sdr_list_n)
    rs_score = np.mean(sdr_list_rs)
    rx_score = np.mean(sdr_list_rx)
    diff_score = np.mean(sdr_list_diff)
    
    return s_score, rs_score, rx_score, diff_score, max_rs, arg_s, arg_n


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
    mu = 255 # 
    s = s * (torch.log(1 + mu*torch.abs(s))/(math.log(1+mu))) # s or tanh(s)
    sr = sr * (torch.log(1 + mu*torch.abs(sr))/(math.log(1+mu))) # s or tanh(s)
    
    return f_SDR(s, sr)