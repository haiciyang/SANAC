from Baseline_0105_Ks import Baseline_0105
import torch
import torch.nn as nn
import torch.nn.functional as F

model = torch.load('../models/0109_170222_32_d1__Base_epoch13.model')
c, c_l = gen_clean_sound(i=1, window_in_data=window_in_data)