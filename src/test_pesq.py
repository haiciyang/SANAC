import os
from utils import *

model = torch.load('../models/0125_164239_32_d1__Base_epoch11.model')

print(calculate_pesq(model, window_in_data=False, soft=False, model_name='test'))