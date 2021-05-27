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
import IPython.display as ipd
from IPython.display import clear_output
from models.Blocks import Bottleneck_new
from models.Prop import Prop as Model
from dataset import Data_TIMIT
from utils import *
import glob
from validation import *

debugging = True

# Won't write out results in file or save models if debugging is True
print('debugging:',debugging)

filters = 100
f2 = 60
d = 1
m = 32
sr = True
lr = 0.0001
br = 8
scale = 1000
label = time.strftime("%m%d_%H%M%S")
target = br/8 if sr else br/16
# br = target * 8 if sr else target * 16
weight_mse = 30
weight_mel = 0.5
weight_qtz = 0.5
weight_etp_total = 0.1
weight_etp_ratio = 0.05
ratio = 1.0
random_data = True
new_mel = False
db = 0
mse_sub_w = [1, 1, 0]
update_ratio = False
batch_size = 100


if new_mel:
    melMSELoss_func = melMSELoss
else:
    melMSELoss_func = melMSELoss_short

window_in_data = False
if window_in_data:
    rebuild_f = rebuild
else:
    rebuild_f = rebuild_extraWindow

traindata = Data_TIMIT('train', overlap = 32, level=db, window=False)
# for i in traindata:
#     print(len(i[0]), len(i[1]))
train_loader = torch.utils.data.DataLoader(traindata, batch_size = 1, shuffle \
                                           = True, num_workers = 1)

testdata = Data_TIMIT('test', overlap = 32, level=db, window=False)
test_loader = torch.utils.data.DataLoader(testdata, batch_size = 1, shuffle \
                                           = True, num_workers = 1)

print('Data loaded Successfully!')

# saved_model = '1008_231358_40_d6_0db'
saved_model = None
finetune = False

if isinstance(saved_model, type(None)) or finetune:
    model_name = '{}_{}_d{}_{}db_{}'.format(label, str(br), str(d), str(db), str(Model.__name__)[:4])
else:
    model_name = saved_model
    
result_path = '../Results/{}.txt'.format(model_name)
model_path = '../save_models/{}.model'.format(model_name)
print('Model Name:', model_name)
hyperpara = ' c_1stDiv {}\n centroid_d {}\n sr {}\n bitrate {}\n lr {}\n weight_mse {}\n weight_mel {}\n weight_qtz {} \n weight_etp_total {} \n weigh_etp_ratio {} \n mse_sub_w {} \n ratio {} \n window_in_data {} \n scale {} \n num_m {} \n random_data {} \n new_mel {} \n udpate_ratio {}'.format(filters, d, sr, br, lr, weight_mse, weight_mel, weight_qtz, weight_etp_total, weight_etp_ratio, mse_sub_w, ratio, window_in_data, scale, m, random_data, new_mel, update_ratio)
print(hyperpara)

# ============ Note ============
if not debugging:
    with open(result_path, 'a') as f:
        f.write('')
# ==============================

if not debugging:
    with open(result_path, 'a') as f:
        f.write(hyperpara + '\n' + 'Model Name:'+ model_name+'\n')
# 
if not isinstance(saved_model, type(None)):
    model = torch.load('../save_models/{}.model'.format(saved_model))
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
    model = Model(block = Bottleneck_new, scale = scale, filters = filters, d_s = d, d_n = d, f2 = f2, num_m = m, sr = sr, ratio = ratio).cuda()
    print('loaded new model:'+ Model.__name__+'\n')

if not debugging:
    with open(result_path, 'a') as f:
        f.write('Model Loaded:'+ Model.__name__ +'\n')

criterion = nn.MSELoss()
if update_ratio:
    optimizer = torch.optim.Adam(list(model.parameters()) + [model.mean_s] + [model.mean_n] + [model.ratio], lr=lr, betas = (0.99, 0.999))
else:
    optimizer = torch.optim.Adam(list(model.parameters()) + [model.mean_s] + [model.mean_n], lr=lr, betas = (0.99, 0.999))
t = 0
print('Model Loaded. Start training...')

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
    if t == 5 and model.stage == 0:
        if not debugging:
            torch.save(model, '../save_models/{}_stage0.model'.format(model_name))
            print('saved model at ../save_models/{}_stage0.model\n'.format(model_name))
            with open(result_path, 'a') as the_file:
                the_file.write('saved at ../save_models/{}_stage0.model\n'.format(model_name))
        model.stage = 1
        model.max_score = 0
        itermax = 0

    qtz_loss = []
    mel_loss = []
    mse_loss = []
    train_etp_cl = []
    train_etp_ns = []
    control = 0
    
    for _, _, source, mixture in train_loader:

        source = source[0].cuda() # shape -> [N, 512]
        mixture = mixture[0].cuda() # shape -> [N, 512]
        noise = (mixture - source).cuda()

        s_h, n_h, prob_s, prob_n = model(mixture, soft = True)  # s_h.shape -> [100, 1, 512]
        s_h = s_h[:,0,:]
        n_h = n_h[:,0,:]
        
        # Calculate losses
        if not isinstance(n_h, type(None)):
#             train_mel = melMSELoss_func(s_h, source) + melMSELoss_func((s_h+n_h), mixture)
            train_mel = melMSELoss_func((s_h+n_h), mixture)
        else:
            train_mel = melMSELoss_func(s_h, mixture)
        mel_loss.append(train_mel.cpu().data.numpy())
             
        entp_cl = None
        entp_ns = None
        loss_qtz = torch.tensor(0).cuda()
        if not isinstance(prob_s, type(None)) and not isinstance(prob_n, type(None)):
            entp_cl = entropy_prob(prob_s)
            train_etp_cl.append(entp_cl.cpu().data.numpy())
            entp_ns = entropy_prob(prob_n)
            train_etp_ns.append(entp_ns.cpu().data.numpy())    
            
            qtz_s = torch.mean(torch.sum(torch.sqrt(prob_s+1e-20), -1) - 1) 
            qtz_n = torch.mean(torch.sum(torch.sqrt(prob_n+1e-20), -1) - 1)
            
            loss_qtz = qtz_s + qtz_n
            
        qtz_loss.append(loss_qtz.cpu().data.numpy())
        
        
        mse_error = mse_sub_w[0]*criterion(n_h+s_h, mixture) + mse_sub_w[1]*criterion(s_h, source) + mse_sub_w[2]*criterion(n_h, noise)
        mse_loss.append(mse_error.cpu().data.numpy())
        
        
        loss = weight_mse * mse_error\
            + weight_qtz * loss_qtz\
            + weight_mel * train_mel
                
        if model.etp == 1:
            control = 1
            loss += weight_etp_total * ((target - entp_cl - entp_ns)**2)\
                + weight_etp_ratio * ((model.ratio - entp_ns/entp_cl)**2)

        
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

#     s_s_score, s_rs_score, s_rx_score, s_diff_score, _, _, _ = test_newData(True)
    h_s_score, h_rs_score, h_rx_score, h_diff_score, pesq_score, arg_s, arg_n = val_prop(False, test_loader, model, window_in_data, debugging, model_name)
    
    entropy = [0, 0]

    if not isinstance(arg_s, type(None)):
        entropy[0] = cal_entropy(arg_s.data.cpu().numpy().flatten())
    if not isinstance(arg_n, type(None)):
        entropy[1] = cal_entropy(arg_n.data.cpu().numpy().flatten())
      
    epoch_list.append((h_s_score, h_rs_score, h_diff_score, entropy[0]))
    

    numbers = '|Test-hard s: {:.2f}, rs: {:.2f}, rx: {:.2f}, diff: {:.2f}, pesq: {:.2f} ||Entropy : {:.2f}, {:.2f}| Train_loss: mse_loss: {:.2f} mel_loss: {:.2f} qtz_loss: {:.2f}\n'\
          .format(h_s_score, h_rs_score, h_rx_score, h_diff_score, pesq_score, entropy[0], entropy[1], np.mean(mse_loss), np.mean(mel_loss), np.mean(qtz_loss))
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
        max_sdr = (h_rs_score, h_rx_score, entropy[0], entropy[1])
        if not debugging:
            torch.save(model, model_path)  
            print('saved at normal model_path')
        itermax = 0
    if finetune:
        itm = 40
        torch.save(model, '../save_models/{}_epoch{}.model'.format(model_name, t))
    if not debugging:
        torch.save(model, '../save_models/{}_epoch{}.model'.format(model_name, t))
        
    
    if model.stage == 1 and model.etp == 0 and itermax >= 3:
        if not debugging:
#             model = torch.load(model_path)        
            torch.save(model, '../save_models/{}_stage1ept0.model'.format(model_name)) 
            print('saved model at ../save_models/{}_stage1ept0.model'.format(model_name))
            with open(result_path, 'a') as the_file:
                the_file.write('saved at ../save_models/{}_stage1ept0.model\n'.format(model_name))
        model.etp = 1
        itermax = 0
        model.max_score = 0
        max_sdr = (0,0,0,0)
