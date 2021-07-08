from torch.utils.data import Dataset
from torchvision import transforms as T 
from config import config
from PIL import Image 
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 
#from .autoaugment import ImageNetPolicy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
import random

def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差
    
    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    
    return data, Gaussian

class CTDataset(Dataset):
    def __init__(self,label_list,transforms=None,train=False,val=False,test=False):
        self.val = val 
        self.train = train 
        self.test = test 
        self.imgs = label_list
        imgs = []
        if self.val:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs 
        elif self.train:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
        elif self.test:
            for index,row in label_list.iterrows():
                imgs.append(row["filename"])
            self.imgs = imgs
        self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.],
                                std = [1.])])

        '''
        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    #T.Normalize(mean = [0.485,0.456,0.406],
                    #            std = [0.229,0.224,0.225])])
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])

            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
#                    Cutout(),
#                    T.RandomHorizontalFlip(),
#                    T.ColorJitter(0.2, 0.2, 0.2),
#                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
#                    ImageNetPolicy(),
#                    T.RandomVerticalFlip(),
#                    T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
                    #T.Normalize(mean = [0, 0, 0],
                    #            std = [1, 1, 1])])
        else:
            self.transforms = transforms
        '''
    def __getitem__(self,index):
        if self.val:
            filename,label = self.imgs[index]
            
            ims = os.listdir(filename)
            ims = [x for x in ims if not '._' in x]
            if len(ims) > config.CT_nums:
                ims.sort(key= lambda x:int(x[:-4]))
                diff = len(ims)/(config.CT_nums+1)
                slices = [ims[int(j*diff)] for j in range(config.CT_nums)]
                #print(slices)
                #_, Gaussian = Gaussian_Distribution(N=1, M=len(ims), sigma=0.1)
                #x = np.linspace(-1,1,len(ims))
                #y = Gaussian.pdf(x)
                #y = y/2.0
                #ims = ims[int(len(ims)/7):int(len(ims)/7*5)]#get lung area
                #slices = random.sample(ims, config.CT_nums) 
            else:
                rnd_indexs =  [random.choice(ims) for i in range(config.CT_nums-len(ims))]
                slices = ims+rnd_indexs
                
            tensor = torch.zeros([config.CT_nums, config.img_weight,config.img_height], dtype=torch.float32)
            for cnt,i in enumerate(slices):
                img = Image.open(filename+'/'+i)
                img = img.convert("L")
                img = self.transforms(img)
                tensor[cnt] = img
            return tensor,label

        elif self.train:
            
            filename,label = self.imgs[index]
            
            ims = os.listdir(filename)
            ims = [x for x in ims if not '._' in x]
            if len(ims) > config.CT_nums:
                ims.sort(key= lambda x:int(x[:-4]))
                diff = len(ims)/(config.CT_nums+1)
                slices = [ims[int(diff*j)] for j in range(config.CT_nums)]
                #print(slices)
                #ims = ims[int(len(ims)/8):int(len(ims)/8*5)]#get lung area
                #slices = random.sample(ims, config.CT_nums) 
            else:
                rnd_indexs =  [random.choice(ims) for j in range(config.CT_nums-len(ims))]
                slices = ims+rnd_indexs
                
            tensor = torch.zeros([config.CT_nums, config.img_weight,config.img_height], dtype=torch.float32)
            for cnt,i in enumerate(slices):
                img = Image.open(filename+'/'+i)
                img = img.convert("L")
                img = self.transforms(img)
                tensor[cnt] = img
            #print(tensor)
            #print(tensor.shape)
            return tensor,label
        elif self.test:
            filename = self.imgs[index]
            
            ims = os.listdir(filename)
            ims = [x for x in ims if not '._' in x]
            if len(ims) > config.CT_nums:
                ims.sort(key= lambda x:int(x[:-4]))
                diff = len(ims)/(config.CT_nums+1)
                slices = [ims[int(diff*j)] for j in range(config.CT_nums)]
                #print(slices)
                #ims = ims[int(len(ims)/8):int(len(ims)/8*5)]#get lung area
                #slices = random.sample(ims, config.CT_nums) 
            else:
                rnd_indexs =  [random.choice(ims) for j in range(config.CT_nums-len(ims))]
                slices = ims+rnd_indexs
                
            tensor = torch.zeros([config.CT_nums, config.img_weight,config.img_height], dtype=torch.float32)
            for cnt,i in enumerate(slices):
                img = Image.open(filename+'/'+i)
                img = img.convert("L")
                img = self.transforms(img)
                tensor[cnt] = img

            return tensor,filename.split('/')[-1]
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

def get_files(root,mode):
    #for val
    if mode == "val":
        df = pd.read_csv('val.csv')
        all_data_path,labels = [],[]
        for i in range(len(df['filename'])):
            all_data_path.append(root+df['filename'][i])
            labels.append(int(df['label'][i]))
        print("loading val dataset")

        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    elif mode == "train": 
        #for train 
        df = pd.read_csv('train.csv')
        all_data_path,labels = [],[]
        for i in range(len(df['filename'])):
            all_data_path.append(root+df['filename'][i])
            labels.append(int(df['label'][i]))
        print("loading train dataset")

        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    elif mode == "test": 
        #for train 
        df = pd.read_csv('test.csv')
        all_data_path = []
        for i in range(len(df['filename'])):
            all_data_path.append(root+df['filename'][i])
            
        print("loading test dataset")

        all_files = pd.DataFrame({"filename":all_data_path})
        return all_files
    else:
        print("check the mode please!")
    
