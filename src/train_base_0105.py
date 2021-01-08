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
from Blocks import BasicBlock, Bottleneck, ChannelChange, Bottleneck_new
# from EntropyControl_curr import AE_control
# from Prop_sr import Prop_Model 
# from Prop_padding import Prop_padding as Model
# from Prop_padding_comp import Prop_padding_comp as Model
# from Prop import Prop as Model
# from Prop_mixture import Prop_mixture as Model
# from Baseline_new import Baseline_new as Model
from Baseline_0105_Ks import Baseline_0105 as Model
from utils import *
import glob

debugging = False
# Won't write out results in file or save models if debugging is True
print('debugging:',debugging)


db = ''
window_in_data = False
if window_in_data:
    rebuild_f = rebuild
else:
    rebuild_f = rebuild_extraWindow

if db == '5db':
    train_path = '../data/half_win_train_ctn_std4_m.pth'
    test_path = '../data/half_win_test_ctn_std4_m.pth'
# elif db == '0db':
#     train_path = '../data/half_win_train_ctn_std_m.pth'
#     test_path = '../data/half_win_test_ctn_std_m.pth'
# elif db == '0db' and window_in_data:
#     train_path = '../data/1116_0db_global_train.pth'
#     test_path = '../data/1116_0db_global_test.pth'
elif db == '0db' and not window_in_data:
    train_path = '../data/1117_0db_global_noWindow_train.pth'
    test_path = '../data/1117_0db_global_noWindow_test.pth'
elif db == '-5db':
    train_path = '../data/half_win_train_ctn_std-5_m.pth'
    test_path = '../data/half_win_test_ctn_std-5_m.pth'
    
train_path = '../data/0105_clean_noWindow_train.pth'
test_path = '../data/0105_clean_noWindow_test.pth'

# Dataset Loading
train_loader = torch.load(train_path)
test_loader = torch.load(test_path)
print('Data loaded Successfully!')

# Model Define

filters = 100
d = 1
f2 = 60
m = 32
sr = True
lr = 0.0001
weight_mse = 30
br = 32
scale = 100
label = time.strftime("%m%d_%H%M%S")
target = br/8 if sr else br/16
# br = target * 8 if sr else target * 16
weight_mel = 5
weight_etp = 1

# saved_model = '1008_231358_40_d6_0db'
saved_model = None
finetune = False

if isinstance(saved_model, type(None)) or finetune:
    model_name = '{}_{}_d{}_{}_{}'.format(label, str(br), str(d), db, str(Model.__name__)[:4])
else:
    model_name = saved_model
    
result_path = '../Results/{}.txt'.format(model_name)
model_path = '../models/{}.model'.format(model_name)
print('Model Name:', model_name)
hyperpara = ' c_1stDiv {}\n centroid_d {}\n sr {}\n bitrate {}\n lr {}\n weight_mse {}\n weight_mel {}\n weight_etp {} \n window_in_data {} \n scale {} \n num_m {}'.format(filters, d, sr, br, lr, weight_mse, weight_mel, weight_etp, window_in_data, scale, m)
print(hyperpara)

# ============ Note ============
if not debugging:
    with open(result_path, 'a') as f:
        f.write('\n')
# ==============================

if not debugging:
    with open(result_path, 'a') as f:
        f.write(hyperpara + '\n' + 'Model Name:'+ model_name+'\n')
# 
if not isinstance(saved_model, type(None)):
    model = torch.load('../models/{}.model'.format(saved_model))
    writeout = 'loaded trained model: {}\n\
    Max score saved in model: {}\n\
    Model_stage: {}\n\
    Model_entropy_control: {}\n'.format(saved_model, model.max_score, model.stage, model.etp)
    print(writeout)
    
    if not debugging:
        with open(result_path, 'a') as f:
            f.write(writeout+'\n')
        
    if finetune:
        print('Finetuning model...')
        if not debugging:
            with open(result_path, 'a') as f:
                f.write('Finetuning model..')
else:
    model = Model(block = Bottleneck_new, scale = scale, filters = filters, d = d,  num_m = m, sr = sr).cuda()
    print('loaded new model:'+ Model.__name__+'\n')

if not debugging:
    with open(result_path, 'a') as f:
        f.write('Model Loaded:'+ Model.__name__ +'\n')

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(list(model.parameters()) + [model.means], lr=lr, betas = (0.99, 0.999))
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
itm = 20

# Training Process

while 1: 
    start = time.time()
    model.train()
    if t == 2 and model.stage == 0:
        if not debugging:
            torch.save(model, '../models/{}_stage0.model'.format(model_name))
            print('saved model at ../models/{}_stage0.model'.format(model_name))
            with open(result_path, 'a') as the_file:
                the_file.write('saved at ../models/{}_stage0.model\n'.format(model_name))
        model.stage = 1
        model.max_score = 0
        itermax = 0
    if model.scale < 500:
        model.scale *= t+1
    train_sdr_s = []
    train_sdr_rs = []
    train_sdr_rx = []
    
    train_etp = []
    control = 0
    
    for wave, inp in train_loader:
        
        inp = inp[0].cuda() #(L, 512)

        s_h, prob = model(inp, soft = True) # s_h - (L, 1, 512)
        s_h = s_h[:,0,:]
        
        rs = rebuild_f(s_h).unsqueeze(dim = 0)

        train_mel_mx = melMSELoss_short(s_h.cpu().data, inp.cpu().data)
            
        entp = None
        if not isinstance(prob, type(None)):
            entp = entropy_prob(prob)
            train_etp.append(entp.cpu().data.numpy())

        loss = weight_mse * criterion(s_h, inp)\
        + weight_mel * train_mel_mx
        
        if model.etp == 1:
            control = 1
            loss += weight_etp * ((target - entp)**2).cuda()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if debugging:
            break
        
    end = time.time()
    Epoch_info = 'epoch_{}| Time:{:.0f} | Control:{:.0f} | Stage: {} | Itermax: {} | etp:{} '.format(t, end-start, control, model.stage, itermax, model.etp)
    print(Epoch_info)
    if not debugging:
        with open(result_path, 'a') as the_file:
            the_file.write(Epoch_info+'\n')

    h_x_score, h_rx_score, max_single_rx, arg = test_base_clean(False, test_loader, model, window_in_data, debugging)
    
    entropy = 0

    if not isinstance(arg, type(None)):
        entropy = cal_entropy(arg.data.cpu().numpy().flatten())
        
    epoch_list.append((h_rx_score, entropy))
    
    numbers = '|Test-hard x: {:.2f} rx: {:.2f} |Entropy :  {:.2f}'\
          .format(h_x_score, h_rx_score, entropy)
    print(numbers)
    if not debugging:
        with open(result_path, 'a') as the_file:
            the_file.write(numbers+'\n')
    
    t += 1
    
#     if model.stage >= 1:
    itermax += 1
#         etp = 1
    if h_rx_score > model.max_score:
        model.max_score = h_rx_score
        max_sdr = (h_rx_score, entropy)
        max_rx = max_single_rx
        itermax = 0
    if not debugging:
        torch.save(model, '../models/{}_epoch{}.model'.format(model_name, t))
#         print('saved at normal model_path')
#         with open(result_path, 'a') as the_file:
#             the_file.write('saved at normal model_path')
    if finetune:
        itm = 40
        torch.save(model, '../models/{}_epoch{}.model'.format(model_name, t))
        
    if model.stage == 0 and itermax >= 5:
        if not debugging:
#             model = torch.load(model_path)
            torch.save(model, '../models/{}_stage0.model'.format(model_name))  
            print('saved model at ../models/{}_stage0.model'.format(model_name))
            with open(result_path, 'a') as the_file:
                the_file.write('saved at ../models/{}_stage0.model\n'.format(model_name))
        model.stage = 1
        model.max_score = 0
        itermax = 0
    
#     if model.stage == 1 and model.etp == 0 and itermax >= 3:
#         if not debugging:
# #             model = torch.load(model_path)        
#             torch.save(model, '../models/{}_stage1ept0.model'.format(model_name)) 
#             print('saved model at ../models/{}_stage1ept0.model'.format(model_name))
#             with open(result_path, 'a') as the_file:
#                 the_file.write('saved at ../models/{}_stage1ept0.model\n'.format(model_name))
#         model.etp = 1
#         itermax = 0
#         model.max_score = 0
#         max_sdr = (0,0,0,0)

#     if model.stage == 1 and itermax >= 2:
#     if model.stage == 1 and itermax >= 5 and model.etp == 1 :
#         if not debugging:
# #             model = torch.load(model_path)
#             torch.save(model, '../models/{}_stage1ept1.model'.format(model_name)) 
#             print('saved model at ../models/{}_stage1ept1.model'.format(model_name))
#             with open(result_path, 'a') as the_file:
#                 the_file.write('saved at ../models/{}_stage1ept1.model\n'.format(model_name))
#         model.stage = 2
# #         itermax = 0
# #         max_sdr = (0,0,0,0)
#         print('Enter stage 2')

    if model.etp == 1 and itermax > itm:
        print('over')
        break