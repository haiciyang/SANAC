from torch.utils import data
from IPython.display import clear_output as clear

class music_data(data.Dataset):
    def __init__(self, filePath)

        label_dict = {
            'P': 0,
            'R': 1,
            'J': 2,
            'G': 3,
            'B': 4,
            'C': 5,
            'A': 6,
            'H': 7,
            'K': 8,
            'E': 9,
            'S': 10,
            'T': 11,
            'L': 12}

        # filePath = '/media/sdc1/Data/ETRI_Music'
        muspath_list = [i for i in os.listdir(filePath) if i.endswith('.wav')]
        label_list = [label_dict.get(i[-7]) for i in muspath_list]
        
        # 1/5 test data; 4/5 training data
        train_index = []
        test_index = []
        for genre in range(13):
            index = [i for i in range(len(label_list)) if label_list[i] == genre]
            perm_index = np.random.permutation(index).tolist()
            test_index += perm_index[:int(len(index)/4)]
            train_index += perm_index[int(len(index)/4):]

        test_index = np.random.permutation(test_index).tolist()
        train_index = np.random.permutation(train_index).tolist()

        train_path = [muspath_list[i] for i in train_index]
        test_path = [muspath_list[i] for i in test_index]
        label_tr = [label_list[i] for i in train_index]
        label_te = [label_list[i] for i in test_index]

        
        # dataloading
        error_file = []


        train_set, test_set = [], []
        i = 0
        
        print(Loading training data:)
        for path in train_path:
            print(Loading training data:)
            clear() 
            print(i)
            i+=1
            try:
                s, sr=librosa.load(filePath+'/'+path, sr=None) 
                trx.append(list(s/np.std(s)))
            except:
#                 print(i, path)
#                 print(label_tr.pop(i))
                error_file.append(path)
                continue

        i = 0
        for path in test_path:
            print(Loading testing data:)
            clear() 
            print(i)
            i+=1
            try:
                s, sr=librosa.load(filePath+'/'+path, sr=None) 
                tex.append(s/np.std(s))
            except:
#                 print(i, path)
#                 print(label_te.pop(i))
                error_file.append(path)
                continue
    
    
    def __len__(self):
        return len(self.mi)
    
    def __getitem__(self, idx):
        return sample = {'trainset': []}
        