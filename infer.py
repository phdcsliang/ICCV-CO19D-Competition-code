import os 
import random 
import time
import json
import torch
import torchvision
import numpy as np 
import pandas as pd 
import warnings
from datetime import datetime
from torch import nn,optim
from torch.optim.optimizer import Optimizer
from config import config 
from collections import OrderedDict
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from dataset.dataloader import *
from sklearn.model_selection import train_test_split,StratifiedKFold
from timeit import default_timer as timer
from pycm import *
#model define 
#from models.model import *
#from models.resnet import *
#from models.rir import *
#from models.attention import *
#from models.senet import *
#from model.new_model import *
from models.ctnet import *

from sklearn.metrics import confusion_matrix

from torchvision import models

#utils
from utils import *
# for comparation
#from cnn_finetune import make_model
#read images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#1. set random.seed and cudnn performance
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#set cuda
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

#get pretrained models
import pretrainedmodels

#usefultools
from torchtoolbox.optimizer import CosineWarmupLr
from torchtoolbox.nn import LabelSmoothingLoss
from torchtoolbox.optimizer import Lookahead
from torchtoolbox.tools import summary

from circle_loss import CircleLoss, NormLinear


#3. test model on public dataset and save the probability matrix
def test(test_loader,model,folds):
    #3.1 confirm the model converted to cuda
    csv_map = OrderedDict({"filename":[],"probability":[]})
    model.cuda()
    model.eval()
    names = []
    labels = []
    with open("./submit/baseline.json","w",encoding="utf-8") as f :
        submit_results = []
        for i,(input,filepath) in enumerate(tqdm(test_loader)):
            #3.2 change everything to cuda and get only basename
            filepath = [os.path.basename(x) for x in filepath]
            with torch.no_grad():
                image_var = Variable(input).cuda()
                #3.3.output
                #print(filepath)
                #print(input,input.shape)
                y_pred,feat = model(image_var)
                #print(y_pred.shape)
                smax = nn.Softmax(1)
                smax_out = smax(y_pred)
            #3.4 save probability to csv files
            csv_map["filename"].extend(filepath)
            
            for output in smax_out:
                prob = ";".join([str(i) for i in output.data.tolist()])
                csv_map["probability"].append(prob)

        result = pd.DataFrame(csv_map)
        result["probability"] = result["probability"].map(lambda x : [float(i) for i in x.split(";")])
        fake_label = []
        fake_name = []
        for index, row in result.iterrows():
            pred_label = np.argmax(row['probability'])+1
            if np.max(row['probability'])>0.9:
                fake_label.append(pred_label)
                fake_name.append(row['filename'])
            if pred_label > 43:
                pred_label = pred_label + 2
            names.append(row['filename'])
            labels.append(pred_label)
            submit_results.append({"FileName":row['filename'],"type":pred_label})
        df = pd.DataFrame({'FileName':names,'type':labels})
        df.to_csv('results.csv',index = False)
        df_fake = pd.DataFrame({'FileName':fake_name,'type':fake_label})
        df_fake.to_csv('fakelist.csv',index = False)
        json.dump(submit_results,f,ensure_ascii=False,cls = MyEncoder)

features_blobs = []

def hook_feature(module, input, output): # input是注册层的输入 output是注册层的输出
    print("hook input",input[0].shape)
    features_blobs.append(output.data.cpu().numpy())
# 对layer4层注册，把layer4层的输出append到features里面

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

#4. more details to build main function    
def main():
    fold = 0
    #4.1 mkdirs
    if not os.path.exists(config.submit):
        os.mkdir(config.submit)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold) + os.sep)
    if not os.path.exists(config.best_models + config.model_name + os.sep +str(fold) + os.sep):
        os.makedirs(config.best_models + config.model_name + os.sep +str(fold) + os.sep)       

    test_only = True

    model = seresnet18(num_classs=config.num_classes)
    model = torch.nn.DataParallel(model)

    #if cuda_avail:
    model.cuda()
    model.eval()
    #4.5.1 read files

    test_files = get_files(config.test_data,"test")

    
    #4.5.4 load dataset

    test_dataloader = DataLoader(CTDataset(test_files,train=False,test=True),batch_size=config.batch_size*2,shuffle=False,pin_memory=False)

    #4.5.5 test
    start = timer()

    if test_only:
        best_model = torch.load(config.best_models +config.model_name+os.sep+ str(fold) +os.sep+ 'model_best.pth.tar')
        model.load_state_dict(best_model["state_dict"])
        test(test_dataloader,model,fold)
    total_loss = 10
    


if __name__ =="__main__":
    main()


