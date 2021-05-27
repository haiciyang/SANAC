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
from utils import *

def val_prop(soft, test_loader, model, window_in_data, debugging, model_name):
    
    with torch.no_grad():
        model.eval()
        
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
            
            
            source = source[0] # [L, 512]
            mixture = mixture[0]
            noise = (mixture-source).cuda()
            source = source.cuda()
            mixture = mixture.cuda()
            s_h, n_h, arg_s, arg_n = model(mixture, soft=soft)
            s_h = s_h[:,0,:]
            n_h = n_h[:,0,:]

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

#             if max_rs_sdr < rs_sdr:
#                 max_rs_sdr = rs_sdr
#                 if not isinstance(n_h, type(None)): 
#                     max_rs = (wave_x, rs, rn)
#                 else:
#                     max_rs = (wave_x, rs)

            if debugging:
                break


        s_score = np.mean(sdr_list_s) 
    #     n_score = np.mean(sdr_list_n)
        rs_score = np.mean(sdr_list_rs)
        rx_score = np.mean(sdr_list_rx)
        pesq_score = calculate_pesq(model, window_in_data, soft, model_name)
        diff_score = np.mean(sdr_list_diff)

    return s_score, rs_score, rx_score, diff_score, pesq_score, arg_s, arg_n

def val_base(soft, test_loader, model, window_in_data, debugging, model_name):

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
        x_h = x_h[:,0,:]
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
