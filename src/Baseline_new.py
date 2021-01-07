import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck, ChannelChange


class Baseline_new(nn.Module):
    def __init__(self, block = None, scale = 1, filters = 40, d = 15, num_m = 32, sr =False):
        
        super(Baseline_new, self).__init__()
        
        self.filters = filters
        self.d = d
        self.num_m = num_m
        self.scale = scale
        self.sr = sr
        self.max_score = 0 # the best score saved by the model
        self.etp = 0 # butten for entropy control 0-off; 1-on
        
        
#         self.mask = torch.rand((d, 512), device = 'cuda:0', requires_grad = True)

        if block.__name__ == 'BasicBlock':
            layers = 3
        elif block.__name__ == 'Bottleneck':
            layers = 3
        
        self.means = torch.rand((self.d, self.num_m), device='cuda', requires_grad = True)
        
        self.initiated = False
        self.stage = 0        
        
        if d <= 5:
            block_d = BasicBlock
        else:
            block_d = Bottleneck

        print(block_d.__name__)
            
        enc_layers = []
        enc_layers.append(block(1, self.filters))
        enc_layers.append(nn.Conv1d(self.filters, self.d, 5, padding = 2))
        if sr == True:
            enc_layers.append(nn.Conv1d(self.d, self.d, 5, padding=2, stride=2))
            enc_layers.append(nn.ReLU())
        for i in range(layers-1):
            enc_layers.append(block_d(self.d, self.d))
        self.enc_base = nn.Sequential(*enc_layers)


        if not sr:
            self.dec_in = nn.Sequential(
                nn.Conv1d(self.d, self.filters//2,  3, padding=1),
                nn.ReLU(),
                block_d(self.filters//2, self.filters//2)
            )
        
        if sr == True:
            self.channel_change = nn.Conv1d(self.filters//2, self.filters, 3, padding=1)
            self.dec_in = nn.Sequential(
                nn.Conv1d(self.d, self.filters, 3, padding=1),
                nn.ReLU(),
                block(self.filters, self.filters)
            )
        dec_layers = []
        for i in range(layers-1):
            dec_layers.append(block(self.filters//2, self.filters//2))
        self.dec_out = nn.Sequential(*dec_layers)
        
        self.fc = nn.Linear(512 * self.filters//2, 512)
        
    
    def forward(self, x, soft=True): # Coding first

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        code = self.enc_base(x)  # 1 -> self.d_s
        
        arg_idx_s = None
        
        if self.stage == 1:
            
            if self.initiated == False:
                self.mean = self.code_init(code, self.d, self.num_m)
                self.initiated = True
            
            code, arg_idx_s = self.code_assign(code, self.mean, soft = soft)
        
            
        s_hat = self.dec_in(code)
        
        if self.sr:
            s_hat = self.sub_pixel(s_hat).cuda()
#             s_hat = self.channel_change(s_hat)
            
        s_hat = self.dec_out(s_hat) # filters//2 
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
                
        return s_hat, arg_idx_s

    
    def forward_sub(self, x, soft=True): # Coding first

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        code = self.enc_base(x)  # 1 -> self.d_s
#         print(x.dtype)
#         code = self.mid_s(x)
        
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(code, self.d_s, self.num_s)
                self.initiated = True
            
            code, arg_idx_s = self.code_assign(code, self.mean_s, soft = soft)
            
            if self.stage == 2:
                
                if not self.sr:
                    code = self.addup_layers_base(code)
                    code_s = code[:, :code.shape[1]//2, :]
                    code_n = code[:, code.shape[1]//2:, :]
                    
                    s_hat = self.dec_2s(code_s) # -- shape (bs, d, 512)
                    n_hat = self.dec_2s(code_n) # -- shape (bs, d, 512)

                    s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                    n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                    s_hat = torch.tanh(self.fc_2s(s_hat))  
                    n_hat = torch.tanh(self.fc_2s(n_hat))
                    
                    return s_hat, n_hat , arg_idx_s, arg_idx_n       

                if self.sr:
                    code = self.addup_sr_in(code)  # d_s -> f2
                    code = self.sub_pixel(code).cuda() # f2 -> f2//2
                    code = self.addup_sr_out_base(code) # f2//2 -> f2

                    code_s = code[:, :code.shape[1]//2, :] # f2 -> f2//2
                    code_n = code[:, code.shape[1]//2:, :]
                    
                    s_hat = self.dec_2s(code_s) # f2//2
                    n_hat = self.dec_2s(code_n) # f2//2

                    s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                    n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                    s_hat = torch.tanh(self.fc_2s(s_hat))  # f2//2 -> 1
                    n_hat = torch.tanh(self.fc_2s(n_hat))
                
                    return s_hat, n_hat , arg_idx_s, arg_idx_n

        if not self.sr:
            s_hat = self.dec_1s(code) # -- shape (bs, d, 512)
            s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
            s_hat = torch.tanh(self.fc_1s(s_hat))

        if self.sr:
            s_hat = self.dec_sr_in(code) # filters
            s_hat = self.sub_pixel(s_hat).cuda()                               
            s_hat = self.dec_sr_out(s_hat) # filters//2 
            s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
            s_hat = torch.tanh(self.fc_sr(s_hat))  
                
        return s_hat, n_hat, arg_idx_s, arg_idx_n


    def code_init(self, codes, d, num_m):
        
        idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
        samples = codes.permute(0, 2, 1).reshape(-1, d)
        
        means = Variable(samples[idx].T, requires_grad = True)
        
        return means
        
    
    def code_assign(self, codes, mean, soft):
        
        # codes shape - (bs, d, L)
        # mean shape - (d, num_m)

        dist_mat = torch.zeros(codes.shape[0], codes.shape[-1], mean.shape[-1]).cuda()   
        # shape (bs, L, num_m)
#         Trade-off between computing speed and high-dimension matrix
        sec = 3
        step = codes.shape[0]//sec
        borders = torch.arange(0, codes.shape[0], step)

        if borders[-1] + step == codes.shape[0]:
            last = torch.Tensor([codes.shape[0]]).type(torch.int64)
            borders = torch.cat((borders, last), dim=0)
        else:
            borders[-1] = codes.shape[0]
            
        for i in range(sec):
            # batch shape (sec, d, L)
            batch = codes[borders[i]:borders[i+1]] 
            mat = torch.sub(batch[:, :, :, None], mean[None, :, None, :]) ** 2 # shape(sec, d, L, num_m)
            mat = torch.sum(mat, dim=1) # shape - (sec, L, num_m)
            dist_mat[borders[i]:borders[i+1]] = mat
        
        # Replace hidden features with means based on probability calculated by softmax

        arg_idx = None
        
        if soft == True:
            # Soft
            prob_mat = F.softmax(- dist_mat * self.scale, dim = -1) # shape(bs, 512, 32)
            x = torch.matmul(prob_mat, mean.transpose(0,1)) # shape(bs, 512, 10)
            x = x.permute(0, 2, 1)
            
            return x, prob_mat
        else:
            # Hard
            arg_idx = torch.argmax(-dist_mat, dim = -1) 
            # arg_idx.shape -> (bs, 512) entry is the index from 0-31
            x = mean[:, arg_idx]  # x.shape -> (10, bs, 512)
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

    
