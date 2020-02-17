import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        
        super(Autoencoder, self).__init__()
        
        self.enc = nn.Sequential(
            nn.Conv1d(1, 5, 3, padding = 1),
            nn.ReLU(),
            nn.Conv1d(5, 10, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(10, 10, 7, padding = 3),
            nn.Tanh()
            # nn.Conv1d(20, 1, 5, padding = 2),  kernel size
            # nn.Tanh()
        )
        
        self.code = torch.rand((10, 32), device='cuda:0', requires_grad = True)
        self.initiated = False
        
        self.dec = nn.Sequential(
            nn.Conv1d(10, 10, 5, padding = 2),
            nn.ReLU(),
            nn.Conv1d(10, 10, 5, padding = 2),
            nn.ReLU()
        )       
        self.fc = nn.Linear(512 * 10, 512)
        

    def forward(self, x):
        
        
        # Encoder
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
#         print('0,20: ',x[0,:,:10])
#         print('40,420: ',x[40,:,400:410])
        x = self.enc(x)  
            # output(N, C, L) -- N = bs, C = 10, L = 512
#         print('0,20-10d: ', x[0,:, :10])
#         print('40,420-10d: ',x[40,:,400:410])

        # Initiate means
        idx_1 = torch.randint(0, x.shape[0], (32,))
        idx_2 = torch.randint(0, x.shape[-1], (1,))
#         if self.initiated == False:
#             self.code = Variable(x[idx_1, :, idx_2].T, requires_grad = True)
#             self.initiated = True
#       
        # Normalize the code and feature-- code.shape (10, 32)
    
#         x = F.normalize(x, dim = 1)
#         self.code.requires_grad = False
#         self.code = F.normalize(self.code, dim = 0)
#         self.code.requires_grad = True

#         x = x.transpose(1,2)  # x.shape - (bs, 512, 10)
#         aff_matrix = torch.matmul(x, self.code) # shape - (bs, 512, 32)
#         argmax = torch.argmax(aff_matrix, dim = 2) # shape - (bs, 512)
#         x = self.code[:, argmax].transpose(0, 1)
#         # transpose (20, bs, 512) - (bs, 10, 512)


        # Calculate the distance   
            # x.shape(bs, 10, 512); code.shape(10, 32)
        mat = torch.sub(x[:,:,:,None], self.code[None,:,None,:]) ** 2
            # mat.shape(bs, 10, 512, 32)
        dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)  dim - the dimension to be reduced
        
        
        # Replace hidden features with codes based on probability calculated by softmax
                
        scale = 1 # Scale the array before for softmax
        prob_mat = F.softmax(- dist_mat * scale, dim = -1) # shape(bs, 512, 32)
        
        x = torch.matmul(prob_mat, self.code.transpose(0,1)) # shape(bs, 512, 10)

        
        # Decoder    
        
        x = self.dec(x.permute(0, 2, 1)) # -- shape (bs, 10, 512)
        x = x.view(-1, x.shape[1] * x.shape[-1])
        x = torch.tanh(self.fc(x))  
        
        
        return x
    
    
    # Same as forward() function except that assign argmax to the mean assignment
    def forward_test(self, x): 
        
        # Encoder
#         print('code: ',self.code.T)
        idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        x = x.view(-1, 1, x.shape[1]) # -- (N, C, L)
        scale = x[idx,:,9]
#         print('0,20: ',x[idx,:,9])
        x = self.enc(x)
        vector = x[idx, :, 9] # (10, 10(dim))
#         print('0,20-10d: ', x[0,:, :10])
#         print('40,420-10d: ',x[40,:,400:410])
            # output(N, C, L) -- N = bs, C = 10, L = 512

        # Calculate the distance   
        mat = torch.sub(x[:,:,:,None], self.code[None,:,None,:]) ** 2
            # mat.shape(bs, 10, 512, 32)
            
        # Experiment scale vs vector
#         print('dis-scale: ', torch.sub(scale, scale.T)**2)
#         print('dis-vec:', torch.sum(torch.sub(vector[:, None, :], vector[None, :,:])**2, dim = -1))
        
        dist_mat = torch.sum(mat, dim = 1) # shape(bs, 512, 32)
#         print(dist_mat.shape)
#         print('dist_mat: ', dist_mat[0, :10, :])
#         print('dist_mat: ', dist_mat[40, 400:410, :])
        # Replace hidden features with codes based on probability calculated by softmax

        arg_idx = torch.argmax(-dist_mat, dim = -1) # arg_idx.shape -> (bs, 512) entry is the index from 0-31
#         print(arg_idx[0][0])
        x = self.code[:, arg_idx]  # x.shape -> (10, bs, 512)
#         print(self.code[:,arg_idx[1][0]])
#         print(x[:,1,0])

        # Decoder    
        
        x = self.dec(x.permute(1, 0, 2)) # target x -- shape (bs, 10, 512)
        x = x.view(-1, x.shape[1] * x.shape[-1])
        x = torch.tanh(self.fc(x))
        
        return x
    
class musClassifier(nn.Module):
    def __init__(self):
        super(musClassifier, self).__init__()

        self.conv1 = nn.Conv1d(1 , 10, 5, stride = 1, dilation = 2)
        self.conv2 = nn.Conv1d(10, 10, 5, stride = 2, dilation = 2)
        self.conv3 = nn.Conv1d(10, 10, 5, stride = 2, dilation = 2)
        #self.conv4 = nn.Conv1d(10, 10, 3, stride = 1, dilation = 2)
        self.fc1 = torch.nn.Linear(2040*10, 512)
        self.fc2 = torch.nn.Linear(512, 13)
        
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

        x = self.ln_con(F.relu(self.conv1(x.view(-1, 1, 8192))))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         print(x.shape)
        x = x.view(-1,2040*10)
        x =self.ln_fc(F.relu(self.fc1(x)))
        output = F.softmax(self.fc2(x))
        
        return output


