import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck, ChannelChange


class Prop_Model(nn.Module):
    def __init__(self, block = None, scale = 1, filters = 40, d_s = 15, d_n = 15, f2 = 50, num_m = 32, \
                 sr =False, ratio = 0.75):
        
        super(Prop_Model, self).__init__()
        
        self.filters = filters
        self.d_s = d_s
        self.d_n = d_n
        self.num_m = num_m
        self.scale = scale
        self.sr = sr
        self.ratio = torch.Tensor([ratio])
        
#         self.mask = torch.rand((d, 512), device = 'cuda:0', requires_grad = True)

        if block.__name__ == 'BasicBlock':
            layers = 3
        elif block.__name__ == 'Bottleneck':
            layers = 3
            
        self.num_s = self.num_m   # number of means gives to source
        self.num_n = self.num_m//3    # number of means gives to noise
        
        self.means_s = torch.rand((self.d_s, self.num_s), device='cuda', requires_grad = True)
        self.means_n = torch.rand((self.d_n, self.num_n), device='cuda', requires_grad = True)
        
        self.initiated = False
        self.stage = 0        
        
        if d_s <= 5:
            block_d = BasicBlock
        else:
            block_d = Bottleneck
        print(block_d.__name__)
            
        enc_layers = []
        enc_layers.append(block(1, self.filters))
        if sr == True:
            enc_layers.append(nn.Conv1d(self.filters, self.filters, 5, padding=2, stride=2))
            enc_layers.append(nn.ReLU())
        for i in range(2):
            enc_layers.append(block(self.filters, self.filters))
        self.enc = nn.Sequential(*enc_layers)

    #             dec_layers = []
        self.mid_s = nn.Sequential(
            block(self.filters//2, self.filters//2),
            nn.Conv1d(self.filters//2, self.d_s, 5, padding=2),
            nn.ReLU(),
            block_d(self.d_s, self.d_s)
        )

        dec_layers = []
        for i in range(2):
            dec_layers.append(block(f2//2, f2//2))
        self.dec_2s = nn.Sequential(*dec_layers)

#         dec_layers = []
#         for i in range(layers-1):
#             dec_layers.append(block(f2//2, f2//2))
#         self.dec_2n = nn.Sequential(*dec_layers)

        if sr == True:

            self.addup_sr_in = nn.Sequential(
                nn.Conv1d(self.d_s, f2, 3, padding=1),
                block(f2, f2),
                block(f2, f2)
            )
            addlayers = []
            for i in range(2):
                addlayers.append(block(f2, f2))
            self.addup_sr_out = nn.Sequential(*addlayers)    
            
        self.fc_2s = nn.Linear(512 * f2//2, 512)
    
    def forward(self, x, soft=True):
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  # 1 -> filters
#         print(x.dtype)
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
        
        code_s = self.mid_s(code_s)  # filters//2 -> d_s
        code_n = self.mid_s(code_n)
        
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(code_s, self.d_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.d_n, self.num_n)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)
   
        code_s = self.addup_sr_in(code_s)  # d_s -> f2
        code_n = self.addup_sr_in(code_n)
                    
        code_s = self.sub_pixel(code_s).cuda() # f2 -> f2//2
        code_n = self.sub_pixel(code_n).cuda()
                    
        code = torch.cat((code_s, code_n), 1)  # f2//2 -> f2
        code = self.addup_sr_out(code)         # f2
        code_s = code[:, :code.shape[1]//2, :] # f2 -> f2//2
        code_n = code[:, code.shape[1]//2:, :]
                    
        s_hat = self.dec_2s(code_s) # f2//2
        n_hat = self.dec_2s(code_n) # 

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc_2s(s_hat))  # f2//2
        n_hat = torch.tanh(self.fc_2s(n_hat))
                    
        return s_hat, n_hat , arg_idx_s, arg_idx_n       
    
    def forward_sub(self, x, soft=True): # Coding first
    # IS first submission function

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  # 1 -> filters
#         print(x.dtype)
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
        
        code_s = self.mid_s(code_s)  # filters//2 -> d_s
        code_n = self.mid_s(code_n)
        
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(code_s, self.d_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.d_n, self.num_n)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)
            
            if self.stage == 2:

                if not self.sr:
                
                    code = torch.cat((code_s, code_n), 1)
                    code = self.addup_layers(code)  # d_s + d_n -> f2
                    code_s = code[:, :code.shape[1]//2, :]
                    code_n = code[:, code.shape[1]//2:, :]
                    
                    s_hat = self.dec_2s(code_s) # f2//2
                    n_hat = self.dec_2s(code_n) # f2//2

                    s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                    n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                    s_hat = torch.tanh(self.fc_2s(s_hat))  # f2//2
                    n_hat = torch.tanh(self.fc_2s(n_hat))  # f2//2
                    
                    return s_hat, n_hat , arg_idx_s, arg_idx_n  

                if self.sr:
                    code_s = self.addup_sr_in(code_s)  # d_s -> f2
                    code_n = self.addup_sr_in(code_n)
                    
                    code_s = self.sub_pixel(code_s).cuda() # f2 -> f2//2
                    code_n = self.sub_pixel(code_n).cuda()
                    
                    code = torch.cat((code_s, code_n), 1)  # f2//2 -> f2
                    code = self.addup_sr_out(code)         # f2
                    code_s = code[:, :code.shape[1]//2, :] # f2 -> f2//2
                    code_n = code[:, code.shape[1]//2:, :]
                    
                    s_hat = self.dec_2s(code_s) # f2//2
                    n_hat = self.dec_2s(code_n) # 

                    s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                    n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                    s_hat = torch.tanh(self.fc_2s(s_hat))  # f2//2
                    n_hat = torch.tanh(self.fc_2s(n_hat))
                    
                    return s_hat, n_hat , arg_idx_s, arg_idx_n     

        if not self.sr:
            s_hat = self.dec_1s(code_s) # d_s 
            n_hat = self.dec_1s(code_n) # d_s

            s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
            n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
            s_hat = torch.tanh(self.fc_1s(s_hat)) # d_s
            n_hat = torch.tanh(self.fc_1s(n_hat)) # d_s

        if self.sr:

            s_hat = self.dec_sr_in(code_s) # filters
            n_hat = self.dec_sr_in(code_n) # 
            
            s_hat = self.sub_pixel(s_hat).cuda() # filters//2
            n_hat = self.sub_pixel(n_hat).cuda() 
                                    
            s_hat = self.dec_sr_out(s_hat) # filters//2
            n_hat = self.dec_sr_out(s_hat) 
                    
            s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
            n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
            s_hat = torch.tanh(self.fc_sr(s_hat))  # filters//2
            n_hat = torch.tanh(self.fc_sr(n_hat))

                
        return s_hat, n_hat , arg_idx_s, arg_idx_n

    
    def forward_sp(self, x, soft = True):
        
        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
#         print(x.dtype)
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
        
        code_s = self.mid_s(code_s)
        code_n = self.mid_s(code_n)
    
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(code_s, self.d_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.d_n, self.num_n)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)

            
            if self.stage == 2:
                
#                 s_hat = self.dec_sr_in(code_s) # -- shape (bs, d//2, 256)
#                 n_hat = self.dec_sr_in(code_n) # -- shape (bs, d//2, 256)

                
#                 code_s = self.sub_pixel(code_s).cuda() #  -- shape (bs, d//4, 512)
#                 code_n = self.sub_pixel(code_n).cuda()
                                  
#                 code = torch.cat((code_s, code_n), 1) # shape (bs, d//2, 512)
                code_s = self.addup_sr_in(code_s)
                code_n = self.addup_sr_in(code_n)
                
                code_s = self.sub_pixel(code_s).cuda()
                code_n = self.sub_pixel(code_n).cuda()
                
                code = torch.cat((code_s, code_n), 1)
                code = self.addup_sr_out(code)
                code_s = code[:, :code.shape[1]//2, :]
                code_n = code[:, code.shape[1]//2:, :]
                
                s_hat = self.dec_2s(code_s) # -- shape (bs, filters//2, 512)
                n_hat = self.dec_2s(code_n) # -- shape (bs, filters//2, 512)

                s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                s_hat = torch.tanh(self.fc_2s(s_hat))  
                n_hat = torch.tanh(self.fc_2s(n_hat))
                
                return s_hat, n_hat , arg_idx_s, arg_idx_n
                

        s_hat = self.dec_sr_in(code_s) # -- shape (bs, d//2, 256)
        n_hat = self.dec_sr_in(code_n) # -- shape (bs, d//2, 256)
        
        s_hat = self.sub_pixel(s_hat).cuda() # (bs, d//4, 512)
        n_hat = self.sub_pixel(n_hat).cuda() # (bs, d//4, 512)
                                
        s_hat = self.dec_sr_out(s_hat) 
        n_hat = self.dec_sr_out(s_hat) 
                
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc_sr(s_hat))  
        n_hat = torch.tanh(self.fc_sr(n_hat))
                
        return s_hat, n_hat , arg_idx_s, arg_idx_n
    
    
    def forward_half(self, x, soft=True): # Coding first

        # Encoder
        
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
#         print(x.dtype)
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
    
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(code_s, self.d_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.d_n, self.num_n)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)

            
            if self.stage == 2:
                
                code = torch.cat((code_s, code_n), 1)
                code = self.addup_layers(code)
                code_s = code[:, :code.shape[1]//2, :]
                code_n = code[:, code.shape[1]//2:, :]
                
                s_hat = self.dec_2(code_s) # -- shape (bs, d, 512)
                n_hat = self.dec_2(code_n) # -- shape (bs, d, 512)

                s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                s_hat = torch.tanh(self.fc_2(s_hat))  
                n_hat = torch.tanh(self.fc_2(n_hat))
                
                return s_hat, n_hat , arg_idx_s, arg_idx_n                

        s_hat = self.dec_1(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec_1(code_n) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc_1(s_hat))  
        n_hat = torch.tanh(self.fc_1(n_hat))
                
        return s_hat, n_hat , arg_idx_s, arg_idx_n


    def code_init(self, codes, d, num_m):
        
        idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
        samples = codes.permute(0, 2, 1).reshape(-1, d)
        
        means = Variable(samples[idx].T, requires_grad = True).cuda()
        
        return means
        
    
    def code_assign(self, codes, mean, soft):
        
        # codes shape - (bs, d, L)
        # mean shape - (d, num_m)

#         dist_mat = torch.zeros(codes.shape[0], codes.shape[-1], mean.shape[-1]).cuda()   
#         # shape (bs, L, num_m)
# #         Trade-off between computing speed and high-dimension matrix
#         sec = 3
#         step = codes.shape[0]//sec
#         borders = torch.arange(0, codes.shape[0], step)

#         if borders[-1] + step == codes.shape[0]:
#             last = torch.Tensor([codes.shape[0]]).type(torch.int64)
#             borders = torch.cat((borders, last), dim=0)
#         else:
#             borders[-1] = codes.shape[0]
            
#         for i in range(sec):
#             # batch shape (sec, d, L)
#             batch = codes[borders[i]:borders[i+1]] 
#             mat = torch.sub(batch[:, :, :, None], mean[None, :, None, :]) ** 2 # shape(sec, d, L, num_m)
#             mat = torch.sum(mat, dim=1) # shape - (sec, L, num_m)
#             dist_mat[borders[i]:borders[i+1]] = mat
        
        mat = torch.sub(codes[:,:,:,None], mean[None,:,None,:]) ** 2
        # mat.shape(bs, d, 512, 32)
        dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
        
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

    