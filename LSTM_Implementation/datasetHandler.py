#Contains the LSTM Dataset class which models the dataset objects which are to be loaded into the PyTorch DataLoaders

import signal
import sys
from os.path import exists
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

def signal_handler(sig, frame):
    print('\nExiting...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class LSTMdataset(Dataset):
    def __init__(self, folder, filename):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        y_data = []
        seqList = []
        seqLengths = []
        with open(folder+"/"+filename, 'r') as d:
            for line in d:
                line = line.split("\n")[0]
                y_data.append(int(line.split(":")[0]))
                seq = self.strToList(line.split(":")[1])
                seqLengths.append(len(seq))
                seqList.append(seq)
                
        self.n_samples = len(y_data)

        # here the first column is the class label, the rest is the frame sequence
        #self.x_data = torch.tensor(seqList, dtype=torch.float32) # size [n_samples, n_time_steps, n_features]
        #self.x_data = self.padData(seqList)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).reshape(-1,1).to(self.device) # size [n_samples, 1]
        self.l_data = torch.tensor(seqLengths).reshape(-1,1).to(self.device)
        
        self.x_data = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in seqList]
        

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.l_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    def strToList(self, st):
        if st == '[]':
            return []
        factor = -1
        for ch in st:
            if ch != '[':
                break
            factor += 1
        if factor == 0:
            return [float(x) for x in st.split("[")[1].split("]")[0].split(", ")]
        
        sList = [x+("]"*factor) if x[len(x) - 1] != ']' else x for x in st[1:len(st)-1].split("]"*factor + ", ")]
        lst = []
        for s in sList:
            lst.append(self.strToList(s))
        return lst
    
    def padData(self, X_list):
        max_len = 0
        num_features = 0
        for seq in X_list:
            if len(seq) != 0:
                num_features = len(seq[0])
            if len(seq) > max_len:
                max_len = len(seq)

        padList = [0]*num_features

        for i in range(len(X_list)):
            iter = max_len - len(X_list[i])
            for j in range(iter):
                X_list[i].append(padList)

        X = torch.tensor(X_list, dtype = torch.float32).to(self.device)

        #print(X)
        return X
