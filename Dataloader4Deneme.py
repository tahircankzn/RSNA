import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

#######################################   2D    ################################################
class data(Dataset):
    def __init__(self):
        data = np.loadtxt('veri_seti_new.csv', delimiter=',')
        self.veriler = data[:, 6:]
        self.hedefler = data[:, 0:5]     # 4800 %80
                                         # 1200 %20
    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        
        
        veri = torch.tensor(self.veriler[idx].reshape(-1, 224, 224)).float()
        hedef = torch.tensor(self.hedefler[idx]).float()


        return veri, hedef

class dataT(Dataset):
    def __init__(self):
        data = np.loadtxt('veri_seti_new_Test.csv', delimiter=',') 
        self.veriler = data[:, 6:]
        self.hedefler = data[:, 0:5]

    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        
        
        veri = torch.tensor(self.veriler[idx].reshape(-1, 224, 224)).float()
        hedef = torch.tensor(self.hedefler[idx]).float()


        return veri, hedef

#######################################   3D    ################################################
class data3D(Dataset):
    def __init__(self):
        data = np.loadtxt('veri_seti_new.csv', delimiter=',')
        self.veriler = data[:4800, 6:]
        self.hedefler = data[:4800, 0:5]     # 4800 %80
                                         # 1200 %20
    def __len__(self):
        return len(self.veriler)

    def __getitem__(self, idx):
        
        
        veri = torch.tensor(self.veriler[idx].reshape(1,224, 224)).float()  # (1,224, 224)
        veri = torch.stack([veri,veri,veri],0)
        hedef = torch.tensor(self.hedefler[idx]).float()

                                                                                     #    [1, 64, 3, 224, 224]
        return veri, hedef                                                           #    [64, 3, 1, 224, 224] olmalı
    
class dataT3D(Dataset):
    def __init__(self):
        data = np.loadtxt('veri_seti_new.csv', delimiter=',') 
        self.veriler = data[4800:, 6:]
        self.hedefler = data[4800:, 0:5]

    def __len__(self):                                   
        return len(self.veriler)

    def __getitem__(self, idx):
        
        
        veri = torch.tensor(self.veriler[idx].reshape(1,224, 224)).float()               
        veri = torch.stack([veri,veri,veri],0)
        hedef = torch.tensor(self.hedefler[idx]).float()


        return veri, hedef
    

