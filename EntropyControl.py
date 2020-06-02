import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck, ChannelChange


class AE_control(nn.Module):
    def __init__(self, block = None, scale = 1, d = 15, filters = 40, num_m = 32, ld = 0.3):
        
        super(AE_control, self).__init__()
        
        self.filters = filters
        self.d = d
        self.num_m = num_m
        self.scale = scale
        self.ld = ld # Want noise to be 30% on the importance of entropy assignments.
#         self.mask = torch.rand((d, 512), device = 'cuda:0', requires_grad = True)

        if block.__name__ == 'BasicBlock':
            layers = 3
        elif block.__name__ == 'Bottleneck':
            layers = 3
            
        self.num_s = self.num_m   # number of means gives to source
        self.num_n = self.num_m//2    # number of means gives to noise
        
        self.means_s = torch.rand((self.d//2, self.num_s), device='cuda', requires_grad = True)
        self.means_n = torch.rand((self.d//2, self.num_n), device='cuda', requires_grad = True)
        
        self.initiated = False
        self.stage = 0

#         N, B, H, P, X, R, C, norm_type, causal = 256, 256, 512, 3, 8, 4, 2, "gLN", False
        
        
#         sep_layers = []
#         for i in range(2):
#             sep_layers.append(block(self.filters, self.filters))
#         sep_layers.append(nn.Sigmoid())
#         self.separator = nn.Sequential(*sep_layers)     
        
#         self.conv_down = ChannelReduce(self.filters//2, self.d)
#         self.conv_up = ChannelReduce(self.d*2, self.filters)
        
        
        if block:
            print(block.__name__)
            
            enc_layers = []
            enc_layers.append(block(1, self.d))
            enc_layers.append(nn.ReLU())
            for i in range(layers):
                enc_layers.append(block(self.d, self.d))

            self.enc = nn.Sequential(*enc_layers)
        

#             dec_layers = []
#             for i in range(layers):
#                 dec_layers.append(block(self.d//2, self.d//2))
#             self.dec = nn.Sequential(*dec_layers)
            
            dec_layers = []
            for i in range(layers):
                dec_layers.append(block(self.d//2, self.d//2))
            self.dec_1 = nn.Sequential(*dec_layers)
            
            dec_layers = []
            for i in range(layers):
                dec_layers.append(block(self.filters//2, self.filters//2))
            self.dec_2 = nn.Sequential(*dec_layers)
        
            addlayers = []
            addlayers.append(nn.Conv1d(self.d, self.filters, 5, padding=2))
            for i in range(3):
                addlayers.append(block(self.filters, self.filters))
            self.addup_layers = nn.Sequential(*addlayers)
            
        self.fc_1 = nn.Linear(512 * self.d//2, 512)
        self.fc_2 = nn.Linear(512 * self.filters//2, 512)
        
    def forward_sep(self,x): # Not using mask
         # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
        
        s = x[:, :x.shape[1]//2, :]
        n = x[:, x.shape[1]//2:, :]
        
        s_hat = self.dec(s) # -- shape (bs, d, 512)
        n_hat = self.dec(n) # -- shape (bs, d, 512)
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
        n_hat = torch.tanh(self.fc(n_hat))  
        
        return s_hat, n_hat
    
    def forward_coding(self,x,soft): # Not using mask
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
        
        arg_idx_s = None
        
        if self.stage == 1:
            
            if self.initiated == False:
                self.mean_s = self.code_init(x, self.num_m)
                self.initiated = True
            
            x, arg_idx_s = self.code_assign(x, self.mean_s, soft = soft)    

        # Decoder    
        
        s_hat = self.dec(x) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat)) 
        
        return s_hat, arg_idx_s
    
    def forward_mask(self, x, soft=True): # Do spearation with mask

        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
#         print(x.dtype)

        mask = self.separator(x)

        code_s = x * mask
        code_n = x * (1 - mask)
        
        arg_idx_s = None
        arg_idx_n = None
        
        # Dimension Reduction
        code_s = F.relu(self.conv_down(code_s))
        code_n = F.relu(self.conv_down(code_n))

        if self.stage == 1:

            if self.initiated == False:
                self.mean_s = self.code_init(code_s, self.num_m)
                self.mean_n = self.code_init(code_n, self.num_m)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)
            
        # Decoder    
        
        s_hat = self.dec(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec(code_n) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
        n_hat = torch.tanh(self.fc(n_hat))  
        
        return s_hat, n_hat , arg_idx_s, arg_idx_n
    
    def forward_c_m(self, x, soft=True): # Coding first

        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
#         print(x.dtype)
        
        arg_idx_s = None
        arg_idx_n = None
        n_hat = None
        
        if self.stage >= 1:
            
            # Dimension Reduction
#             x = F.relu(self.conv_down(x))
            
            if self.initiated == False:
                self.mean_s = self.code_init(x, self.num_m)
                self.initiated = True
            
            x, arg_idx_s = self.code_assign(x, self.mean_s, soft = soft)
            
#             return x,  n_hat , arg_idx_s, arg_idx_n
        
        # Decoder    
        
            if self.stage == 2:
                
                x = F.relu(self.conv_up(x))
                
                mask = self.separator(x)
                code_s = x * mask
                code_n = x * (1 - mask)

                s_hat = self.dec_2(code_s) # -- shape (bs, d, 512)
                n_hat = self.dec_2(code_n) # -- shape (bs, d, 512)

                s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                s_hat = torch.tanh(self.fc_2(s_hat))  
                n_hat = torch.tanh(self.fc_2(n_hat))
                
                return s_hat, n_hat , arg_idx_s, arg_idx_n
        
        x = self.dec_1(x) # -- shape (bs, d, 512)
        x = x.view(-1, x.shape[1] * x.shape[-1])
        x = torch.tanh(self.fc_1(x))

        return x, n_hat , arg_idx_s, arg_idx_n
    
    
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
                self.mean_s = self.code_init(code_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.num_n)
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
            
#                 code_s = self.addup_layers(code_s)
#                 code_ss = code_s[:, :code_s.shape[1]//2, :]
#                 code_sn = code_s[:, code_s.shape[1]//2:, :]
            
#                 code_ss = self.dec_2(code_ss)
#                 code_sn = self.dec_2(code_sn)
#                 code_n = self.dec(code_n)
#                 code_ss = code_ss.view(-1, code_ss.shape[1] * code_ss.shape[-1])
#                 code_sn = code_sn.view(-1, code_sn.shape[1] * code_sn.shape[-1])
#                 code_n = code_n.view(-1, code_n.shape[1] * code_n.shape[-1])
                
#                 code_ss = torch.tanh(self.fc_2(code_ss))  
#                 code_sn = torch.tanh(self.fc_2(code_sn))
#                 code_n = torch.tanh(self.fc(code_n))
                
#                 return code_ss, code_sn+code_n, arg_idx_s, arg_idx_n
                

        s_hat = self.dec_1(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec_1(code_n) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc_1(s_hat))  
        n_hat = torch.tanh(self.fc_1(n_hat))
                
        return s_hat, n_hat , arg_idx_s, arg_idx_n
    
    def forward_sp(self, x, soft = True):
        
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
                self.mean_s = self.code_init(code_s, self.num_s)
                self.mean_n = self.code_init(code_n, self.num_n)
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
    
    def forward_half_2(self, x, soft=True): # Coding first

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
                self.mean_s = self.code_init(code_s, self.num_m)
                self.mean_n = self.code_init(code_n, self.num_m)
                self.initiated = True
            
            code_s, arg_idx_s = self.code_assign(code_s, self.mean_s, soft = soft)
            code_n, arg_idx_n = self.code_assign(code_n, self.mean_n, soft = soft)

            
            if self.stage == 2:
                
                code_s = self.addup_layers_1(code_s)
                
                code_ss = code_s[:, :code_s.shape[1]//2, :]
                code_sn = code_s[:, code_s.shape[1]//2:, :]
                
                code_n = code_n+code_sn
                
                s_hat = self.dec(code_ss) # -- shape (bs, d, 512)
                n_hat = self.dec(code_n) # -- shape (bs, d, 512)

                s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
                n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
                s_hat = torch.tanh(self.fc(s_hat))  
                n_hat = torch.tanh(self.fc(n_hat))
                
                return s_hat, n_hat , arg_idx_s, arg_idx_n                

        s_hat = self.dec(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec(code_n) # -- shape (bs, d, 512)

        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
        n_hat = torch.tanh(self.fc(n_hat))
                
        return s_hat, n_hat , arg_idx_s, arg_idx_n
    
    def code_init(self, codes, num_m):
        
        idx = torch.randint(0, codes.shape[0] * codes.shape[-1], (num_m,))
        samples = codes.permute(0, 2, 1).reshape(-1, self.d//2)
        
        means = Variable(samples[idx].T, requires_grad = True)
        
        return means
        
    
    def code_assign(self, codes, mean, soft):
        
        # codes shape - (bs, d, L)
        # mean shape - (d, num_m)
        
#         mat = torch.sub(codes[:,:,:,None], mean[None,:,None,:]) ** 2

#         # mat.shape(bs, d, 512, 32)
#         dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
        
        dist_mat = torch.zeros(codes.shape[0], codes.shape[-1], mean.shape[-1]).cuda()   # shape (bs, L, num_m)
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
    
       
    # For test, the score of plain-AE
    def test_no_qtz(self, x): 
        
        # Encoder

        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)
        mask = self.separator(x)
        code_s = x * mask
        code_n = x * (1 - mask)
            # output(N, C, L) -- N = bs, C = 10, L = 512

        # Decoder    
        
        s_hat = self.dec(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec(code_n) # -- shape (bs, d, 512)
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
        n_hat = torch.tanh(self.fc(n_hat)) 
        
        
        return s_hat, n_hat
    
    def forward_old(self, x, s, n):
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        s = s.view(-1, 1, s.shape[1]) # -- (N, C, L)
        n = n.view(-1, 1, n.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
        s = self.enc(s)
        n = self.enc(n)
            # output(N, C, L) -- N = bs, C = d, L = 512

        
        if self.stage == 1:
            
            # Calculate the distance   
            
            if self.initiated == False:
                idx = torch.randint(0, x.shape[0] * x.shape[-1], (self.num_m,))
                samples = x.permute(0, 2, 1).reshape(-1, self.d)

                self.means = Variable(samples[idx].T, requires_grad = True)
                self.initiated = True

#             mat = torch.sub(x[:,:,:,None], self.means[None,:,None,:]) ** 2
            mat_s = torch.sub(s[:,:,:,None], self.means[None,:,None,:]) ** 2
            mat_n = torch.sub(n[:,:,:,None], self.means[None,:,None,:]) ** 2
        
            # mat.shape(bs, d, 512, 32)
            dist_mat_s = torch.sum(mat_s, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
            dist_mat_n = torch.sum(mat_n, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
            
            dist_mat = dist_mat_s + self.ld*dist_mat_n
            
            # Replace hidden features with codes based on probability calculated by softmax

            if self.soft == True:
                # Soft
                prob_mat = F.softmax(- dist_mat * self.scale, dim = -1) # shape(bs, 512, 32)

                x = torch.matmul(prob_mat, self.means.transpose(0,1)) # shape(bs, 512, 10)
                x = x.permute(0, 2, 1)
            else:
                # Hard
                arg_idx = torch.argmax(-dist_mat, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
                x = self.means[:, arg_idx]  # x.shape -> (10, bs, 512)
                x = x.permute(1, 0, 2)

        
        # Decoder    
        
        x = self.dec(x) # -- shape (bs, d, 512)
        x = x.view(-1, x.shape[1] * x.shape[-1])
        x = torch.tanh(self.fc(x))  
        
        
        return x
    
       


# class AE_convtas



    
class plain_Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv1d(1 , 10, 3, padding = 1)
        self.conv2 = nn.Conv1d(10, 10, 5, padding = 2)
        self.conv3 = nn.Conv1d(10, 10, 5, padding = 2)
        #self.conv4 = nn.Conv1d(10, 10, 3, stride = 1, dilation = 2)
        self.fc1 = torch.nn.Linear(512*10, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
#         self.fc2 = torch.nn.Linear(512, 13)
        
        self.dropout =nn.Dropout(0.2)
        
        self.ln_con = nn.LayerNorm(8184)
        self.ln_fc = nn.LayerNorm(512)
        
#         nn.init.kaiming_normal_(self.conv1.weight)
#         nn.init.kaiming_normal_(self.conv2.weight)
#         nn.init.kaiming_normal_(self.conv3.weight)
#         nn.init.kaiming_normal_(self.conv4.weight)
#         nn.init.xavier_normal_(self.fc1.weight)
#         nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x):

#         x = self.ln_con(F.relu(self.conv1(x.view(-1, 1, 8192))))
#         x = self.dropout(x)
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         print(x.shape)
        x = x.view(-1,x.shape[1]*x.shape[2])
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
#         output = F.softmax(self.fc2(x))
        
        return x
    
    
