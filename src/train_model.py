import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TZ'] = 'EST+05EDT,M4.1.0,M10.5.0'
# Importing the libraries 
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
time.tzset()
# from Models import Autoencoder
# from TestData import SingleMus
from Data_TIMIT import Data_TIMIT
import IPython.display as ipd
from IPython.display import clear_output
from Blocks import BasicBlock, Bottleneck, ChannelChange
# from EntropyControl_curr import AE_control
# from Prop_sr import Prop_Model 
# from Prop_padding import Prop_padding as Model
# from Prop_padding_comp import Prop_padding_comp as Model
from Prop import Prop as Model
from utils import *
import glob

debugging = False
# Won't write out results in file or save models if debugging is True
print('debugging:',debugging)


db = '5db'

if db == '5db':
    train_path = '../data/half_win_train_ctn_std4_m.pth'
    test_path = '../data/half_win_test_ctn_std4_m.pth'
elif db == '0db':
    train_path = '../data/half_win_train_ctn_std_m.pth'
    test_path = '../data/half_win_test_ctn_std_m.pth'

# Dataset Loading
train_loader = torch.load(train_path)
test_loader = torch.load(test_path)
print('Data loaded Successfully!')

# Model Define

filters = 30
d = 6
f2 = 60
m = 160
sr = True
lr = 0.0001
weight1 = 1/20
weight2 = 1/60
target = 6
ratio = 1/3
label = time.strftime("%m%d_%H%M%S")
br = target * 8 if sr else target * 16
mel_weight = 1/12

model_name = None

if isinstance(model_name, type(None)):
    model_name = '{}_{}_d{}_{}'.format(label, str(br), str(d), db)
    
result_path = '../Results/{}.txt'.format(model_name)
model_path = '../models/{}.model'.format(model_name)
print('Model Name:', model_name)
hyperpara = ' c_1stDiv {}\n c_2ndDiv {}\n centroid_d {}\n sr {}\n bitrate {}\n lr {}\n weight1 {}\n weight2 {}\n ratio {}\n mel_weight {}\n'.format(filters, f2, d, sr, br, lr, weight1, weight2, ratio, mel_weight)
print(hyperpara)

# ======= Special Note =========
if not debugging:
    with open(result_path, 'a') as f:
        f.write('Mel Loss; on mix and clean\n')
# ==============================

if not debugging:
    with open(result_path, 'a') as f:
        f.write(hyperpara + '\n' + 'Model Name:'+ model_name+'\n')
# 
if model_path in glob.glob('../models/*.model'):
    model = torch.load(model_path)
    print('loaded trained model: {}\n\
    Max score saved in model: {}\n\
    Model_stage: {}\n\
    Model_entropy_control: {}\n'.format(model_name, model.max_score, model.stage, model.etp))
else:
    model = Model(block = Bottleneck, scale = 10, filters = filters, d_s = d, d_n = d, f2 = f2, num_m = m, sr = sr, ratio = ratio).cuda()
    print('loaded new model:'+ Model.__name__+'\n')

if not debugging:
    with open(result_path, 'a') as f:
        f.write('Model Loaded:'+ Model.__name__ +'\n')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()) + [model.means_s] + [model.means_n] + [model.ratio], lr=lr)
t = 0
print('Model Loaded. Start training...')

# model.stage = 1
# model.max_score = 0
#
epochs = 100
epoch_list = []
max_sdr = (0,0,0,0)
max_rs = None
flct = 0.1
# etp = 0
itermax = 0


# Training Process

while 1: 
    start = time.time()
    model.train()
    if t == 3 and model.stage == 0:
        model.stage = 1
    if model.scale < 500:
        model.scale *= t+1
    train_sdr_s = []
    train_sdr_rs = []
    train_sdr_rx = []
    
    train_etp_cl = []
    train_etp_ns = []
    control = 0
    
    for wave_s, wave_x, source, mixture in train_loader:
        
        source = source[0]
        mixture = mixture[0]
        
        noise = (mixture - source).cuda()
        source = source.cuda()
        mixture = mixture.cuda()

        s_h, n_h, prob_s, prob_n = model(mixture, soft = True)
        
#         train_sdr_s.append(SDR(s_h, source))
#         rs = rebuild(s_h).unsqueeze(dim = 0)
       
#         train_sdr_rs.append(SISDR(wave_s, rs))
#         rn = rebuild(n_h).unsqueeze(dim = 0)
#         train_sdr_rx.append(SISDR(wave_x, rn+rs))
        train_mel_cl = melMSELoss_short(s_h.cpu().data, source.cpu().data)
        train_mel_mx = melMSELoss_short((s_h+n_h).cpu().data, mixture.cpu().data)
        
        entp_cl = None
        entp_ns = None
        if not isinstance(prob_s, type(None)) and not isinstance(prob_n, type(None)):
            entp_cl = entropy_prob(prob_s)
            train_etp_cl.append(entp_cl.cpu().data.numpy())
            entp_ns = entropy_prob(prob_n)
            train_etp_ns.append(entp_ns.cpu().data.numpy())    
            
#         if model.stage == 1:
#         loss = mulaw_loss(s_h, source, criterion) + mulaw_loss(n_h+s_h, mixture, criterion)
#         if model.stage != 2:
        loss = criterion(s_h, source) + criterion(n_h+s_h, mixture)\
        + mel_weight * train_mel_cl.cuda() \
        + mel_weight * train_mel_mx.cuda()
        
#         if model.stage == 2:
#             loss = criterion(s_h, mixture)
#             - SDR(wave_s, rs, cuda = True) - 1/10*SDR(wave_x, rs+rn, cuda = True)\
# #             + 1/50 * train_mel
# #             - SDR(wave_x-wave_s, rn, cuda = True)\
           
#         else:
#             loss = - SDR(wave_s, rs, cuda = True) - 1/10*SDR(wave_x, rs+rn, cuda = True) \
# #             + 1/50 * train_mel  
# #             - SDR(wave_x-wave_s, rn, cuda = True)
        
        if model.etp == 1:
            control = 1
            loss += weight1 * ((target - entp_cl - entp_ns)**2) \
                + weight2 * ((ratio - entp_ns/entp_cl)**2)
            
#             if abs(target_clean - entp_cl) > flct:
# #                 print('c1')
#                 control  = 1
#                 loss += weight * ((target_clean - entp_cl) ** 2)
#             if abs(target_noise - entp_ns) > flct:
# #                 print('c2')
#                 control = 1
#                 loss += weight * ((target_noise - entp_ns) ** 2)
        
#         if etp == 1:
#             if abs(target - entp_cl - entp_ns) > flct:
#                 control = 1
#                 loss += 1/2 * ((target - entp_cl - entp_ns) ** 2).cpu()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
#         break
        

    end = time.time()
    Epoch_info = 'epoch_{}| Time:{:.0f} | Control:{:.0f} | Stage: {} | Itermax: {} | etp:{} '.format(t, end-start, control, model.stage, itermax, model.etp)
    print(Epoch_info)
    if not debugging:
        with open(result_path, 'a') as the_file:
            the_file.write(Epoch_info+'\n')

#     s_s_score, s_rs_score, s_rx_score, s_diff_score, _, _, _ = test_newData(True)
    h_s_score, h_rs_score, h_rx_score, h_diff_score, max_single_rs,  arg_s, arg_n = test_newData(False, test_loader, model)
    
    entropy = [0, 0]

    if not isinstance(arg_s, type(None)):
        entropy[0] = cal_entropy(arg_s.data.cpu().numpy().flatten())
    if not isinstance(arg_n, type(None)):
        entropy[1] = cal_entropy(arg_n.data.cpu().numpy().flatten())
      
    epoch_list.append((h_s_score, h_rs_score, h_diff_score, entropy[0]))
    
    numbers = '|Test-hard s: {:.2f}, rs: {:.2f}, rx: {:.2f}, diff: {:.2f} \n\
          |Entropy : {:.2f}, {:.2f}'\
          .format(h_s_score, h_rs_score, h_rx_score, h_diff_score, 
                  entropy[0], entropy[1])
 #          |Test-soft s: {:.2f}, rs: {:.2f}, rx: {:.2f}, diff: {:.2f} \n\
    print(numbers)
    if not debugging:
        with open(result_path, 'a') as the_file:
            the_file.write(numbers+'\n')
    
    t += 1
    if model.stage >= 1:
        itermax += 1
#         etp = 1
        if h_rx_score > model.max_score:
            model.max_score = h_rx_score
            max_sdr = (h_rs_score, h_rx_score, entropy[0], entropy[1])
            max_rs = max_single_rs
            if not debugging:
                torch.save(model, model_path)  
            itermax = 0
    
    if model.stage == 1 and model.etp == 0 and itermax >= 3:
        model.etp = 1
        itermax = 0
        model.max_score = 0
        max_sdr = (0,0,0,0)

#     if model.stage == 1 and itermax >= 2:
    if model.stage == 1 and itermax >= 5 and model.etp == 1 :
        model.stage = 2  
#         itermax = 0
#         max_sdr = (0,0,0,0)
        print('Enter stage 2')

    if model.etp == 1 and itermax > 20:
        print('over')
        break
