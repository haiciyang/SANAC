import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
       
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(inplanes, planes, 3, padding = 1)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(planes, planes, 5, padding = 2)
#         print(inplanes,planes)

    def forward(self, x):
        
        identity = x
        out = self.conv1(x)
#         out = self.bn(out)
        out = self.relu(out)
#         identity = out
        out = self.conv2(out)
#         out = self.bn(out)
        out = self.relu(out)
        out += identity
#         out = self.relu(out)
    
#         print(out.shape)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(Bottleneck, self).__init__()

        width = outplanes//3
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_in = nn.Conv1d(inplanes, width, 7, padding = 3)
        self.conv_bt = nn.Conv1d(width, width, 5, padding = 2)
        self.conv_out = nn.Conv1d(width, outplanes, 3, padding = 1)
        
        self.bn_bt = nn.BatchNorm1d(width)
        self.bn_out = nn.BatchNorm1d(outplanes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
#         print('hh')
#         print(x.dtype)
        identity = x
#         print(x.dtype)
#         fake()s
        out = self.conv_in(x)
#         out = self.bn_bt(out)
        out = self.relu(out)

        out = self.conv_bt(out)
#         out = self.bn_bt(out)
        out = self.relu(out)

        out = self.conv_bt(out)
#         out = self.bn_bt(out)
        out = self.relu(out)
        
        out = self.conv_out(out)
#         out = self.bn_out(out)
        out = self.relu(out)
        out += identity
#         out = self.relu(out)

        return out

class Bottleneck_new(nn.Module):
    
    def __init__(self, in_plane, dilation=1):
        super(Bottleneck_new, self).__init__()
        
        kernel_size = 9
        width = in_plane//4
        padding_size = (dilation * (kernel_size-1))//2
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_in = nn.Conv1d(in_plane, width, kernel_size, padding = padding_size, dilation=dilation)
        self.conv_bt = nn.Conv1d(width, width, kernel_size, padding = padding_size, dilation=dilation)
        self.conv_out = nn.Conv1d(width, in_plane, kernel_size, padding = padding_size, dilation=dilation)
        
#         self.bn_bt = nn.BatchNorm1d(width)
#         self.bn_out = nn.BatchNorm1d(outplanes)
        
        self.relu = nn.ReLU()

    def forward(self, x):

        identity = x

        out = self.conv_in(x)
        out = self.relu(out)

        out = self.conv_bt(out)
        out = self.relu(out)
        
        out = self.conv_out(out)
        out = self.relu(out)
        out += identity

        return out

class ChannelChange(nn.Module):
    expansion = 4

    def __init__(self, inplanes, outplanes):
        super(ChannelChange, self).__init__()

#         width = outplanes//2
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_in = nn.Conv1d(inplanes, outplanes, 9, padding = 4)
        self.conv_out = nn.Conv1d(outplanes, outplanes,9, padding = 4)
        
        self.conv_change = nn.Conv1d(inplanes, outplanes, 9, padding = 4)
        
#         self.bn_bt = nn.BatchNorm1d(width)
#         self.bn_out = nn.BatchNorm1d(outplanes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
#         print('hh')
#         print(x.dtype)
        identity = self.conv_change(x)
#         print(x.dtype)
#         fake()s
        out = self.conv_in(x)
#         out = self.bn_bt(out)
        out = self.relu(out)

        out = self.conv_out(out)
#         out = self.bn_bt(out)
        out = self.relu(out)

        out = self.conv_out(out)
#         out = self.bn_bt(out)
        out = self.relu(out)

        out += identity
#         out = self.relu(out)

        return out