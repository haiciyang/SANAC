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

def melMSELoss(s, sr, n_mels = [8, 16, 32, 128]): # input waveform(torch.Tensor)
    
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
#         m = mse(s_mel, sr_mel)
        loss += error
#         print(error)
    return loss/len(n_mels)

def melMSELoss_0109(s, sr, n_mels = [8, 16, 32, 128]): 
    
    assert s.shape[1] == 512
    assert sr.shape[1] == 512
    loss = 0
    eps = 1e-20
    mse = nn.MSELoss().cuda()
    s = s.cuda()
    sr = sr.cuda()
    for n in n_mels:
        melspec = torchaudio.transforms.MelSpectrogram(n_fft=512, n_mels=n).cuda()
        s_mel = torch.mean(melspec(s), -1)
        sr_mel = torch.mean(melspec(sr), -1)
        # s_mel.shape -> [bt, n]
        s_mel = torch.log(s_mel + eps)
        sr_mel = torch.log(sr_mel + eps)
        mse_score = 0
        for i in range(len(s_mel)):
            error = mse(s_mel[i], sr_mel[i])
            mse_score += error
#         m = mse(s_mel, sr_mel)
        loss += mse_score
    return loss/len(n_mels) 

def l1Loss(s, sr): # input waveform(torch.Tensor)
    
    length = min(len(s),len(sr))
    s = s[:length]
    sr = sr[:length]
    
    loss = torch.sum(torch.abs(s-sr))/length
    

    return loss


def rebuild(output, overlap = 64):
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
    
    window = np.hamming(overlap*2) 
    window = np.concatenate((window[:overlap],np.ones(512-overlap*2),window[overlap:]))
    window = window.reshape(1,-1)
    window = window.astype(np.float32)
    
    output *= window
    wave = np.zeros(len_wav)
    for i in range(len(output)):
#         print(i)
        wave[i*(512-overlap):i*(512-overlap)+512] += output[i]
    
    return torch.tensor(wave, dtype=torch.float).requires_grad_()

def test_newData(soft, test_loader, model, window_in_data, debugging):
    
    with torch.no_grad():
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
        if window_in_data:
            rebuild_f = rebuild
        else:
            rebuild_f = rebuild_extraWindow

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


            rs = rebuild_f(s_h).unsqueeze(dim = 0)
            rs_sdr = SISDR(wave_s, rs)
            if not isinstance(n_h, type(None)):
                rn = rebuild_f(n_h).unsqueeze(dim = 0)
                rx_sdr = SISDR(wave_x, rs+rn)
            else:
                rx_sdr = SISDR(wave_x, rs)


            sdr_list_rs.append(rs_sdr)
            sdr_list_rx.append(rx_sdr)

            start_sdr = SISDR(wave_s, wave_x/max(wave_x[0]))
            sdr_list_diff.append(rs_sdr-start_sdr)

            if max_rs_sdr < rs_sdr:
                max_rs_sdr = rs_sdr
                if not isinstance(n_h, type(None)): 
                    max_rs = (wave_x, rs, rn)
                else:
                    max_rs = (wave_x, rs)

            if debugging:
                break


        s_score = np.mean(sdr_list_s) 
    #     n_score = np.mean(sdr_list_n)
        rs_score = np.mean(sdr_list_rs)
        rx_score = np.mean(sdr_list_rx)
        diff_score = np.mean(sdr_list_diff)

    return s_score, rs_score, rx_score, diff_score, max_rs, arg_s, arg_n

def test_base(soft, test_loader, model, window_in_data, debugging):

    x_sdr_list = []
    rx_sdr_list = []
    max_rx_sdr = 0
    max_rx = None
    
    if window_in_data:
        rebuild_f = rebuild
    else:
        rebuild_f = rebuild_extraWindow

    i = 0
    for wave_s, wave_x, source, mixture in test_loader: 

        source = source[0]
        mixture = mixture[0]
#         print(data)
        noise = (mixture-source).cuda()
        source = source.cuda()
        mixture = mixture.cuda()
        x_h, arg = model(mixture, soft=soft)
        
        x_sdr_list.append(SISDR(mixture, x_h))
        
        rx = rebuild_f(x_h).unsqueeze(dim = 0)
        rx_sdr = SISDR(wave_x, rx)

        rx_sdr_list.append(rx_sdr)
        
        if max_rx_sdr < rx_sdr:
            max_rx_sdr = rx_sdr
            max_rx = (wave_x, rx)

#         if debugging:
#             break
    
    rx_score = np.mean(rx_sdr_list)
    x_score = np.mean(x_sdr_list)
    
    return  x_score, rx_score, max_rx, arg

def test_base_clean(soft, test_loader, model, window_in_data, debugging):

    x_sdr_list = []
    rx_sdr_list = []
    max_rx_sdr = 0
    max_rx = None
    
    if window_in_data:
        rebuild_f = rebuild
    else:
        rebuild_f = rebuild_extraWindow

    i = 0
    for wave, inp in test_loader: 

        inp = inp[0].cuda()
        x_h, arg = model(inp, soft=soft)
        x_h = x_h[:, 0, :]
        
        x_sdr_list.append(SISDR(inp, x_h))
        
        rx = rebuild_f(x_h).unsqueeze(dim = 0)
        rx_sdr = SISDR(wave, rx)

        rx_sdr_list.append(rx_sdr)
        
        if max_rx_sdr < rx_sdr:
            max_rx_sdr = rx_sdr
            max_rx = (wave, rx)

        if debugging:
            break
    
    rx_score = np.mean(rx_sdr_list)
    x_score = np.mean(x_sdr_list)
    
    return  x_score, rx_score, max_rx, arg

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