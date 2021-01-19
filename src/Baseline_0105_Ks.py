import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck, ChannelChange, Bottleneck_new


class Baseline_0105(nn.Module):
    def __init__(self, block = Bottleneck_new, scale = 1, filters = 40, d = 15, num_m = 32, sr =False):
        
        super(Baseline_0105, self).__init__()
        
        self.d = d
        self.num_m = num_m
        self.scale = scale
        self.sr = sr
        self.max_score = 0 # the besdt score saved by the model
        self.etp = 0 # butten for entropy control 0-off; 1-on
        self.means = torch.arange(-1, 1, 2/self.num_m)[None, :].cuda().requires_grad_()
        self.atv_f = nn.LeakyReLU()

#         self.means = None
        self.initiated = False
        self.stage = 0  

#         print(block_d.__name__)
         
        # ======== Encoder =========
        enc_layers = []
        enc_layers.append(nn.Conv1d(1, filters, 9, padding=4))
        enc_layers.append(self.atv_f)
        # ----- 1st bottleneck -----
        enc_layers.append(block(in_plane=filters, dilation=1))
        enc_layers.append(block(filters, dilation=2))
        if sr == True:
            enc_layers.append(nn.Conv1d(filters, filters, 9, padding=4, stride=2))
            enc_layers.append(self.atv_f)
        # ----- 2nd bottleneck -----
        enc_layers.append(block(filters, dilation=1))
        enc_layers.append(block(filters, dilation=2))
        enc_layers.append(nn.Conv1d(filters, self.d, 9, padding=4))
        enc_layers.append(nn.Tanh())
#         enc_layers.append(self.atv_f)

        self.enc_base = nn.Sequential(*enc_layers)
        
        
        # ======== Decoder =========
        self.dec_in = nn.Sequential(
                nn.Conv1d(self.d, filters, 9, padding=4),
                self.atv_f,
                block(filters, dilation=1),
                block(filters, dilation=2)
        )
        
        filters2 = filters//2 if sr else filters
        self.dec_out = nn.Sequential(
            block(filters2, dilation=1),
            block(filters2, dilation=2),
            nn.Conv1d(filters2, 1, 9, padding=4)#,
#             nn.Tanh()
        )
        
    
    def forward(self, x, soft=True): # Coding first

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        code = self.enc_base(x)  # 1 -> self.d_s
        
        # Quantization
        arg_idx_s = None
        if self.stage == 1:
#             if self.initiated == False:
#                 self.means = self.code_init(code, self.d, self.num_m)
#                 self.initiated = True
            code, arg_idx_s = self.code_assign(code, self.means, soft = soft)
        # Decoder
        s_hat = self.dec_in(code)
        if self.sr:
            s_hat = self.sub_pixel(s_hat).cuda()
        s_hat = self.dec_out(s_hat) # filters//2  
                
        return s_hat, arg_idx_s


    def code_init(self, codes, d, num_m):
        
        # codes.shape - (bt, d, L)
        if self.d == 1:
            means = torch.arange(-1, 1, 2/self.num_m)[None, :].cuda().requires_grad_()
        else:
            idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
            samples = codes.permute(0, 2, 1).reshape(-1, d) # (bt*L, d)
            means = Variable(samples[idx].T, requires_grad = True) # (d, num_m)

#         idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
#         samples = codes.permute(0, 2, 1).reshape(-1, d) # (bt*L, d)

#         means = Variable(samples[idx].T, requires_grad = True) # (d, num_m)
        
        return means
        
    
    def code_assign(self, codes, means, soft):
        
        # codes shape - (bs, d, L)
        # means shape - (d, num_m)
        
        mat = torch.sub(codes[:,:,:,None], means[None,:,None,:]) ** 2 # (bs, d, L, num_m)
#         print(mat.shape)
#         print(codes[60])
#         print(codes[60,:,0], mat[60,:,0,:])
#         print(codes[60,:,10], mat[60,:,10,:])
#         print(codes[60,:,20], mat[60,:,20,:])
        dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)
        
        # Replace hidden features with means based on probability calculated by softmax

        arg_idx = None
        
        if soft == True:
            # Soft
            eps = 1e-20
            prob_mat = F.softmax(- dist_mat*self.scale, dim = -1) # shape(bs, 512, num_m)
            x = torch.matmul(prob_mat, means.transpose(0,1)) # shape(bs, 512, d)
            x = x.permute(0, 2, 1)
            
            return x, prob_mat
        
        else:
            # Hard
            arg_idx = torch.argmax(- dist_mat, dim = -1) 
            # arg_idx.shape -> (bs, 512) entry is the index from 0-31
            x = means[:, arg_idx]  # x.shape -> (10, bs, 512)
            x = x.permute(1, 0, 2)
        
            # arg_idx is only used for entropy calculating when doing hard argmax in test
            return x, arg_idx
                                  
    def sub_pixel(self, x):  
        
        # x.shape - (bs, self.d, L)
        # output.shape - (bs, self.d//2, L*2)
        
        bs = x.shape[0]
        d = x.shape[1] 
        L = x.shape[2]
        up_x = torch.zeros(bs, d//2, L*2).cuda()
        for i in range(0, d//2):
            x_sub = x[:, i*2:(i+1)*2, :] # (bs, 2, L)
            up_x[:,i,:] = x_sub.transpose(1,2).reshape(bs, 1, L*2)[:,0,:]
            
        return up_x

    
