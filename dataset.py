import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler
from utils import *

def truncate_data(data, cut):
    N, t, ndim = data.shape # N x t x ndim

    data = data.transpose(0, 2, 1) # N x ndim x t 

    if cut == -1:
        move_steps = 1
    else:
        move_steps = int(t/cut)

    list_ = []
    for i in range(N):
        list_j = []
        for j in range(move_steps,t+1,move_steps):
            list_k = []
            for k in range(ndim):
                temp = np.pad( data[i][k][:j], (t-j,0), mode='mean')
                list_k.append(temp) # ndim x j
            list_k = np.asarray(list_k).transpose(1,0) # t x ndim

            list_j.append(list_k) #cut x t x ndim
        list_j = np.asarray(list_j)
        list_.append(list_j)

    return list_ 

class PHM_DATA_Challenge_2018(Dataset):
    def __init__(self, args, train):
        self.type = args.type
        self.train = train
        self.dataset = args.dataset
        self.cut = args.cut

        name2len = {'01hr':int(1*3600/4), '03hr':int(3*3600/4), '06hr':int(6*3600/4), '12hr':int(12*3600/4), '24hr':int(24*3600/4)}
        if '03hr' in self.dataset:
            self.ntimesteps = name2len['03hr']
            self.dataset_front = '03hr'
        elif '06hr' in self.dataset:
            self.ntimesteps = name2len['06hr']
            self.dataset_front = '06hr'
        elif '12hr' in self.dataset:
            self.ntimesteps = name2len['12hr']
            self.dataset_front = '12hr'
        elif '24hr' in self.dataset:
            self.ntimesteps = name2len['24hr']
            self.dataset_front = '24hr'
        elif '01hr' in self.dataset:
            self.ntimesteps = name2len['01hr']
            self.dataset_front = '01hr'

        if len(self.dataset)>4 and (self.dataset[4] in ['1', '2', '3', '4', '5']):
            self.csv_name = self.dataset[:5]
        else:
            self.csv_name = self.dataset[:4]

        self.data, self.labels, self.N_FEATURES = self.generateDataset()

        self.N_CLASSES = len(np.unique(self.labels))

        self.N = self.ntimesteps

        print("Finish loading dataset PHM_DATA_Challenge_2018")
        print("Featrues: ", self.N_FEATURES, "Total: ", self.N)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix], self.labels[ix], ix

    def generateDataset(self):
        DATASET_PATH = '/mnt/data3/nfs/ting/phm_data_challenge_2018/preprocessed/'
        print(self.csv_name)
        data_list = pd.read_csv(DATASET_PATH+self.csv_name+'.csv')
        data_list = data_list.loc[ data_list['train'] == self.train ]
        #if running the dataset with fixed earliness
        real_ntimesteps = self.ntimesteps
        if "sub" in self.type:
            idx = self.type.find("sub") + 3
            sub = float(self.type[idx:idx+4]) #EX: 0.95
            real_ntimesteps = self.ntimesteps
            self.ntimesteps = int(self.ntimesteps * sub)
            print("Using sub data with {} proportions and its length is {}".format(str(sub),str(self.ntimesteps)))

        #if doning over-sampling in balance way
        if "up" in self.type and "%" in self.type and self.train == 'train':

            anomaly_num = data_list.loc[ data_list['label'] == True ].shape[0]
            normal_num = data_list.loc[ data_list['label'] == False ].shape[0]
            idx_ = self.type.find("up") + 2

            prop = float(self.type[idx_:idx_+2]) * 0.01
            append_time_up = int( (((normal_num/(1-prop))-normal_num))/anomaly_num )
            print("Over-samlping with the factor {} to match the percentage {}.".format(str(append_time_up),prop))

        data = []
        labels = []
        machine = []

        ##### Read data part
        for index, x in tqdm(data_list.iterrows()):
            df = pd.read_csv(DATASET_PATH+self.dataset_front+'/'+x['machine']+'/'+x['file'])

            #if running the dataset we fixed earliness
            if "sub" in self.type:
                df= df.head(self.ntimesteps)
            

            sample = df.drop(columns=['time', 'Tool', 'stage', 'Lot', 
                'runnum', 'recipe', 'recipe_step']).fillna(method='ffill',axis=1).to_numpy()  # Length x nFeatures

            #if using over-sampling
            if "up" in self.type and "%" in self.type and self.train == 'train':
                apt = append_time_up
            else:
                apt = 1

            for _ in range(apt):
                data.append(sample)
                labels.append(x['label'])
                machine.append(x['machine'])

        data = np.asarray(data)
        labels = np.asarray(labels)

        print(data.shape, labels.shape)

        #Truncate Data
        data = truncate_data(data, self.cut)  #From: N x t x ndim, To: N x t/cut x t x ndim

        data = torch.tensor(np.asarray(data).astype(np.float32), dtype=torch.float)  
        labels = torch.tensor(np.array(labels).astype(np.int32), dtype=torch.float)
            
        return data, labels, data.shape[-1]


def load_data(args, without_shuffle = False):

    data = PHM_DATA_Challenge_2018(args, train='train')
    test_data = PHM_DATA_Challenge_2018(args, train='test')

    train_sampler = SubsetRandomSampler(np.arange(len(data)))
    test_sampler = SubsetRandomSampler(np.arange(len(test_data)))

    if without_shuffle:
        print("Don't shuffle the data")
        train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=args.batch_size,
                                               drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=args.batch_size,
                                                drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               drop_last=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=args.batch_size,
                                                sampler=test_sampler,
                                                drop_last=False)
        
    return train_loader, test_loader, data, test_data