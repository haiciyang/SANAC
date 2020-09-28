import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Blocks import BasicBlock, Bottleneck


class Denoising(nn.Module):

    def __init__(self, block = None, f = 15, d = 100):
        
        super(Denoising, self).__init__()
        
        self.d = d
        self.stage = 0

        if block.__name__ == 'BasicBlock':
            layers = 3
        elif block.__name__ == 'Bottleneck':
            layers = 3
   
        enc_layers = []
        enc_layers.append(nn.Conv1d(1, self.d, 55, padding = 27))
        enc_layers.append(nn.ReLU())
#         enc_layers.append(block(1, self.d))
        for i in range(layers):
            enc_layers.append(block(self.d, self.d))

        self.enc = nn.Sequential(*enc_layers)
        
        sep_layers = []
        for i in range(3):
            sep_layers.append(block(self.d, self.d))
        sep_layers.append(nn.Sigmoid())
        self.separator = nn.Sequential(*sep_layers) 
        
        
        add_layers = []
        for i in range(2):
            sep_layers.append(block(self.d, self.d))
        self.addup_layers = nn.Sequential(*add_layers) 
        
        dec_layers = []
        dec_layers.append(block(self.d//2, self.d//2))
        dec_layers.append(nn.Conv1d(self.d//2, f, 11, padding=5))
        for i in range(2):
            dec_layers.append(block(f, f))
        self.dec1 = nn.Sequential(*dec_layers)
        
        dec_layers = []
        dec_layers.append(block(self.d//5, self.d//5))
        dec_layers.append(nn.Conv1d(self.d//5, f, 11, padding=5))
        for i in range(2):
            dec_layers.append(block(f, f))
        self.dec2 = nn.Sequential(*dec_layers)
        
        self.fc1 = nn.Linear(512 * f, 512)
        self.fc2 = nn.Linear(512 * f, 512)
       
        

    def forward_pure(self,x): # Not using mask
         # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
        
        s_hat = self.dec(x) # -- shape (bs, d, 512)
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        s_hat = torch.tanh(self.fc(s_hat))  
        
        return s_hat
    
    
    def forward_mask_long(self, x, soft=True): # Do spearation with mask

        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
#         print(x.dtype)
        mask = self.separator(x)
#         print(mask.shape)
#         fake()
        code_s = x * mask
        code_n = x * (1 - mask)
        
        # Decoder    
        
        s_hat = self.dec1(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec2(code_n) # -- shape (bs, d, 512)
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc1(s_hat))
        n_hat = torch.tanh(self.fc2(n_hat))
        
#         s_hat = s_hat.view(s_hat.shape[0], s_hat.shape[2])
#         n_hat = n_hat.view(n_hat.shape[0], n_hat.shape[2])
        
        return s_hat, n_hat #, arg_idx_s, arg_idx_n

    
    def forward_half(self, x, soft=True): # Do spearation with mask

        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
        
        code_s = x[:, :x.shape[1]//2, :]
        code_n = x[:, x.shape[1]//2:, :]
        
        if self.stage == 1:
            code = torch.cat((code_s, code_n), 1)
            code = self.addup_layers(code)  # d_s + d_n -> f2
            code_s = code[:, :code.shape[1]//2, :]
            code_n = code[:, code.shape[1]//2:, :]
                    
#                     s_hat = self.dec_2s(code_s) # f2//2
#                     n_hat = self.dec_2s(code_n) # f2//2

#                     s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
#                     n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
#                     s_hat = torch.tanh(self.fc_2s(s_hat))  # f2//2
#                     n_hat = torch.tanh(self.fc_2s(n_hat))  # f2//2
                    
#                     return s_hat, n_hat , arg_idx_s, arg_idx_n  
        # Decoder    
        
        s_hat = self.dec1(code_s) # -- shape (bs, d, 512)
        n_hat = self.dec1(code_n) # -- shape (bs, d, 512)
        s_hat = s_hat.view(-1, s_hat.shape[1] * s_hat.shape[-1])
        n_hat = n_hat.view(-1, n_hat.shape[1] * n_hat.shape[-1])
        s_hat = torch.tanh(self.fc1(s_hat))
        n_hat = torch.tanh(self.fc1(n_hat))
        
#         s_hat = s_hat.view(s_hat.shape[0], s_hat.shape[2])
#         n_hat = n_hat.view(n_hat.shape[0], n_hat.shape[2])
        
        return s_hat, n_hat

class Sep_Autoencoder(nn.Module):
    def __init__(self, block = None, d = 16, num_m = 32, soft = True):
        
        super(Sep_Autoencoder, self).__init__()
        
        self.d = d
        self.num_m = num_m
        self.scale = 1
        self.soft = soft

        if block.__name__ == 'BasicBlock':
            layers = 4
        elif block.__name__ == 'Bottleneck':
            layers = 3
  
        
#         self.code = torch.rand((self.d, self.num_m), device='cuda:0', requires_grad = True)
        self.initiated = False
        self.stage = 0
        
        
        self.enc = nn.Sequential(
            
            nn.Conv1d(1, 5, 3, padding = 1),
#             nn.BatchNorm1d(5),
            nn.ReLU(),
            
            nn.Conv1d(5, 10, 5, padding = 2),
#             nn.BatchNorm1d(10),
            nn.ReLU(),
            
            nn.Conv1d(10, self.d, 7, padding = 3),
#             nn.BatchNorm1d(self.d),
            nn.Tanh()
            # nn.Conv1d(20, 1, 5, padding = 2),  kernel size
            # nn.Tanh()
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv1d(self.d//2, 5, 7, padding = 3),
            nn.ReLU(),
            nn.Conv1d(5, 10, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(10, self.d//2, 3, padding = 1),
            nn.ReLU()
        )       
        
        self.dec2 = nn.Sequential(
            nn.Conv1d(self.d//2, 5, 7, padding = 3),
            nn.ReLU(),
            nn.Conv1d(5, 10, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(10, self.d//2, 3, padding = 1),
            nn.ReLU()
        )  
        
        if block:
            print('run block')
            
            enc_layers = []
            enc_layers.append(block(1, self.d))
            for i in range(layers):
                enc_layers.append(block(self.d, self.d))

            self.enc = nn.Sequential(*enc_layers)
        

            dec_layers = []
            for i in range(layers):
                dec_layers.append(block(self.d//2, self.d//2))

            self.dec1 = nn.Sequential(*dec_layers)
            self.dec2 = nn.Sequential(*dec_layers)
    
        self.fc = nn.Linear(512 * self.d//2, 512)
       
        

    def forward(self, x):
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)  
            # output(N, C, L) -- N = bs, C = d, L = 512

        x1 = x[:, :x.shape[1]//2, :]
        x2 = x[:, x.shape[1]//2:, :]
        
        if self.stage == 1:
            
            # Calculate the distance   
            
            if self.initiated == False:
                idx = torch.randint(0, x.shape[0] * x.shape[-1], (self.num_m,))
                samples1 = x1.permute(0, 2, 1).reshape(-1, self.d//2)
                samples2 = x2.permute(0, 2, 1).reshape(-1, self.d//2)
                samples = torch.cat((samples1, samples2), dim = 0)

                self.code = Variable(samples[idx].T, requires_grad = True)
                self.initiated = True

            mat1 = torch.sub(x1[:,:,:,None], self.code[None,:,None,:]) ** 2
            mat2 = torch.sub(x2[:,:,:,None], self.code[None,:,None,:]) ** 2
            
            # mat.shape(bs, 10, 512, 32)
            dist_mat1 = torch.sum(mat1, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
            dist_mat2 = torch.sum(mat2, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced

            
            # Replace hidden features with codes based on probability calculated by softmax

            if self.soft == True:
                # Soft
                prob_mat1 = F.softmax(- dist_mat1 * self.scale, dim = -1) # shape(bs, 512, 32)
                prob_mat2 = F.softmax(- dist_mat2 * self.scale, dim = -1) # shape(bs, 512, 32)

                x1 = torch.matmul(prob_mat1, self.code.transpose(0,1)) # shape(bs, 512, 10)
                x2 = torch.matmul(prob_mat2, self.code.transpose(0,1)) # shape(bs, 512, 10)
                x1 = x1.permute(0, 2, 1)
                x2 = x2.permute(0, 2, 1)
            else:
                # Hard
                arg_idx1 = torch.argmax(-dist_mat1, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
                arg_idx2 = torch.argmax(-dist_mat2, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
                x1 = self.code[:, arg_idx1]  # x.shape -> (10, bs, 512)
                x2 = self.code[:, arg_idx2]  # x.shape -> (10, bs, 512)
                x1 = x1.permute(1, 0, 2)
                x2 = x2.permute(1, 0, 2)

        
        # Decoder    
        
        
        x1 = self.dec1(x1) # -- shape (bs, d, 512)
        x2 = self.dec2(x2) # -- shape (bs, d, 512)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[-1])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[-1])
        x1 = torch.tanh(self.fc(x1))  
        x2 = torch.tanh(self.fc(x2))     
        
        return x1, x2
    
       
    # For test, the score of quantizaiton
    def test_soft_qtz(self, x): 
          
        # Encoder

        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)
        x1 = x[:, :x.shape[1]//2, :]
        x2 = x[:, x.shape[1]//2:, :]
            # output(N, C, L) -- N = bs, C = 10, L = 512

        # Calculate the distance   
        mat1 = torch.sub(x1[:,:,:,None], self.code[None,:,None,:]) ** 2
        mat2 = torch.sub(x2[:,:,:,None], self.code[None,:,None,:]) ** 2
            
            # mat.shape(bs, 10, 512, 32)
        dist_mat1 = torch.sum(mat1, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
        dist_mat2 = torch.sum(mat2, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced

        
        # Replace hidden features with codes based on probability calculated by softmax

#         #  Hard
#         arg_idx = torch.argmax(-dist_mat, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
#         x = self.code[:, arg_idx]  # x.shape -> (10, bs, 512)
#         x = x.permute(1, 0, 2)
    
        # Soft
        prob_mat1 = F.softmax(- dist_mat1 * self.scale, dim = -1) # shape(bs, 512, 32)
        prob_mat2 = F.softmax(- dist_mat2 * self.scale, dim = -1) # shape(bs, 512, 32)

        x1 = torch.matmul(prob_mat1, self.code.transpose(0,1)) # shape(bs, 512, 10)
        x2 = torch.matmul(prob_mat2, self.code.transpose(0,1)) # shape(bs, 512, 10)
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        

        # Decoder    
        
        x1 = self.dec1(x1) # -- shape (bs, d, 512)
        x2 = self.dec2(x2) # -- shape (bs, d, 512)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[-1])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[-1])
        x1 = torch.tanh(self.fc(x1))  
        x2 = torch.tanh(self.fc(x2))
        
        return x1, x2
    
    
     # For test, the score of quantizaiton
    def test_hard_qtz(self, x): 
          
        # Encoder

        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)
        x1 = x[:, :x.shape[1]//2, :]
        x2 = x[:, x.shape[1]//2:, :]
            # output(N, C, L) -- N = bs, C = 10, L = 512

        # Calculate the distance   
        mat1 = torch.sub(x1[:,:,:,None], self.code[None,:,None,:]) ** 2
        mat2 = torch.sub(x2[:,:,:,None], self.code[None,:,None,:]) ** 2
            
            # mat.shape(bs, 10, 512, 32)
        dist_mat1 = torch.sum(mat1, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
        dist_mat2 = torch.sum(mat2, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced

        
        # Replace hidden features with codes based on probability calculated by softmax

        #  Hard
        arg_idx1 = torch.argmax(-dist_mat1, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
        arg_idx2 = torch.argmax(-dist_mat2, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
        x1 = self.code[:, arg_idx1]  # x.shape -> (10, bs, 512)
        x2 = self.code[:, arg_idx2]  # x.shape -> (10, bs, 512)
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
    
        # Decoder    
        
        x1 = self.dec1(x1) # -- shape (bs, d, 512)
        x2 = self.dec2(x2) # -- shape (bs, d, 512)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[-1])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[-1])
        x1 = torch.tanh(self.fc(x1))  
        x2 = torch.tanh(self.fc(x2))
        
        return x1, x2, arg_idx1, arg_idx2
    
       
    # For test, the score of plain-AE
    def test_no_qtz(self, x): 
        
        # Encoder

        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        x = self.enc(x)
            # output(N, C, L) -- N = bs, C = 10, L = 512
        x1 = x[:, :x.shape[1]//2, :]
        x2 = x[:, x.shape[1]//2:, :]

        # Decoder    
        
        x1 = self.dec1(x1) # -- shape (bs, d, 512)
        x2 = self.dec2(x2) # -- shape (bs, d, 512)
        x1 = x1.view(-1, x1.shape[1] * x1.shape[-1])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[-1])
        x1 = torch.tanh(self.fc(x1))  
        x2 = torch.tanh(self.fc(x2))
        
        return x1, x2



    
    
