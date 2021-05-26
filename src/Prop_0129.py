import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import Bottleneck_new


class Prop_0129(nn.Module):
    def __init__(self, block = Bottleneck_new, scale = 1, filters = 40, d_s = 15, d_n = 15, f2 = 50, num_m = 32, \
                 sr =False, ratio = 1/3):
        
        super(Prop_0129, self).__init__()
        
        self.d = d_s
        self.num_m = num_m
        self.scale = scale
        self.sr = sr
        self.max_score = 0 # the besdt score saved by the model
        self.etp = 0 # butten for entropy control 0-off; 1-on
        self.atv_f = nn.LeakyReLU()
        self.ratio = torch.tensor(ratio).cuda()

        self.initiated = False
        self.stage = 0  
        
        self.mean_s = torch.arange(-1, 1, 2/self.num_m)[None, :].cuda().requires_grad_()
        self.mean_n = torch.arange(-1, 1, 2/self.num_m)[None, :].cuda().requires_grad_()

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
        self.enc_1 = nn.Sequential(*enc_layers)

        # ----- 2nd bottleneck -----
        self.enc_2 = nn.Sequential(
            block(filters//2, dilation=1),
            block(filters//2, dilation=2),
            nn.Conv1d(filters//2, self.d, 9, padding=4),
            nn.Tanh()
        )       

        # ======== Decoder for mixture =========

        self.dec_in = nn.Sequential(
                nn.Conv1d(self.d, filters, 9, padding=4),
                self.atv_f,
                block(filters, dilation=1),
                block(filters, dilation=2)
        )
        
        self.upsample_layer = nn.Sequential(
            nn.Conv1d(filters, filters, 9, padding=4),
            self.atv_f
        )

        filters2 = filters//2 if sr else filters
        self.dec_out = nn.Sequential(
            block(filters2, dilation=1),
            block(filters2, dilation=2),
            nn.Conv1d(filters2, 1, 9, padding=4)#,
#             nn.Tanh()
        )
        
        # ========== Addup layer =========
        
        self.add_up = nn.Sequential(
            block(filters*2, dilation=1),
            block(filters*2, dilation=2)
        )
    
    def forward(self, x, soft=True): # Coding first

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc_1(x)  # 1 -> filters
#         print(x.dtype)
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
        
        code_s = self.enc_2(code_s)  # filters//2 -> d_s
        code_n = self.enc_2(code_n)
        
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft) 

        s_hat = self.dec_in(code_s) # d_s -> filters
        n_hat = self.dec_in(code_n) # 
        
        if self.stage == 2:

                code = torch.cat((s_hat, n_hat), 1)  # filters -> 2*filters
                code = self.add_up(code)         # 2*filters
                s_hat = code[:, :code.shape[1]//2, :] # 2*filters -> filters
                n_hat = code[:, code.shape[1]//2:, :]
                
        if self.sr:
            
            s_hat = self.upsample_layer(s_hat).cuda()
            n_hat = self.upsample_layer(n_hat).cuda()
            s_hat = self.sub_pixel(s_hat).cuda() 
            n_hat = self.sub_pixel(n_hat).cuda() 
                                    
        s_hat = self.dec_out(s_hat) # filters
        n_hat = self.dec_out(n_hat) # filters
                
        return s_hat, n_hat , arg_idx_s, arg_idx_n

    

    def code_init(self, codes, d, num_m):
        
        idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
        samples = codes.permute(0, 2, 1).reshape(-1, d)
        
        means = Variable(samples[idx].T, requires_grad = True).cuda()
        
        return means
        
    
    def code_assign(self, codes, means, soft):
        
        # codes shape - (bs, d, L)
        # means shape - (d, num_m)
        
        mat = torch.sub(codes[:,:,:,None], means[None,:,None,:]) ** 2 # (bs, d, L, num_m)

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
        # x.shape -> (B, C, L)
        bs = x.shape[0]
        C = x.shape[1] 
        L = x.shape[2]
        n = 2

        x = x.permute(0,2,1)
        x = x.reshape(-1, L, C//n, n)
        x = x.permute(0,1,3,2)
        x = x.reshape(-1, L*2, C//n)
        x = x.permute(0,2,1)

        return x

    
