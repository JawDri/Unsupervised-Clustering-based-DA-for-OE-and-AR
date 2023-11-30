import os
import shutil
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from utils.folder import ImageFolder
import numpy as np
import pandas as pd
import cv2


class PytorchDataSet(Dataset):
    
    def __init__(self, df):
        FEATURES = list(i for i in df.columns if i!= 'labels')
        TARGET = "labels"

        from sklearn.preprocessing import StandardScaler
        Normarizescaler = StandardScaler()
        Normarizescaler.fit(np.array(df[FEATURES]))
        
        # for test data, In test data, it's easier to fill it with something on purpose.
        
        if "labels" not in df.columns:
            df["labels"] = 9999
        
        self.df = df
        #self.indx = df.index.values
        
        self.train_X = np.array(self.df[FEATURES])
        self.train_Y = np.array(self.df[TARGET])
        self.train_X = Normarizescaler.transform(self.train_X)
        
        
        self.train_X = torch.from_numpy(self.train_X).float()
        self.train_Y = torch.from_numpy(self.train_Y).long()

        self.imgs = self.train_X
        self.tgts = self.train_Y

        
    
    def __len__(self):
        
        return len(self.df)

    
    
    def __getitem__(self, idx):
        
        
        return self.imgs[idx].view(1, 9), self.tgts[idx], idx##########




def generate_dataloader(args):

    Source_train = pd.read_csv("/content/drive/MyDrive/SRDC/data/Source_train.csv")
    
    Source_train = PytorchDataSet(Source_train)
    
    

    Source_test = pd.read_csv("/content/drive/MyDrive/SRDC/data/Source_test.csv")
    #print(Source_test.index.values)
    Source_test = PytorchDataSet(Source_test)

    Target_train = pd.read_csv("/content/drive/MyDrive/SRDC/data/Target_train.csv")
    Target_train = PytorchDataSet(Target_train)


    Target_test = pd.read_csv("/content/drive/MyDrive/SRDC/data/Target_test.csv")
    Target_test = PytorchDataSet(Target_test)
      
    
    
       
    source_train_dataset = Source_train
    source_test_dataset = Source_test

    target_train_dataset = Target_train

    target_test_dataset = Target_test
    target_test_dataset_t = Target_test
    
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_test_loader_t = torch.utils.data.DataLoader(
        target_test_dataset_t, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return source_train_loader, target_train_loader, target_test_loader, target_test_loader_t, source_test_loader




 
 
 
