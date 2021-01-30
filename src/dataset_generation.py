from Data_noiseTIMIT_random import Data_noiseTIMIT_random as dataModel
import torch

data = dataModel('train', overlap = 32, level=0, window=False)
train_loader = torch.utils.data.DataLoader(data, batch_size = 100, shuffle \
                                           = True, num_workers = 4)
torch.save(train_loader, '../data/0129_0db_noWindow_random_train.pth')
