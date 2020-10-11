import torch
from Data_TIMIT import Data_TIMIT

overlap = 64
data = Data_TIMIT('test', mix_num = 1, overlap = overlap, level=-5)
test_loader = torch.utils.data.DataLoader(data, batch_size = 1, shuffle \
                                           = True, num_workers = 4)
torch.save(test_loader, '../data/half_win_train_ctn_std-5_m.pth')



data = Data_TIMIT('train', mix_num = 1, overlap = 64, level=-5)
train_loader = torch.utils.data.DataLoader(data, batch_size = 50, shuffle \
                                           = True, num_workers = 4)
torch.save(train_loader, '../data/half_win_train_ctn_std-5_m.pth.pth')