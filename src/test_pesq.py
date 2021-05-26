import os
from utils import *

model = torch.load('../models/0201_190536_16_d1_0db_Prop.model')

print(calculate_pesq(model, window_in_data=False, soft=False, model_name='test'))