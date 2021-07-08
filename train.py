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


'''
class Ralamb(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(Ralamb, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Ralamb, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Ralamb does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, radam_step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        radam_step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        radam_step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = radam_step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                radam_step = p_data_fp32.clone()
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    radam_step.addcdiv_(-radam_step_size * group['lr'], exp_avg, denom)
                else:
                    radam_step.add_(-radam_step_size * group['lr'], exp_avg)

                radam_norm = radam_step.pow(2).sum().sqrt()
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                if weight_norm == 0 or radam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / radam_norm

                state['weight_norm'] = weight_norm
                state['adam_norm'] = radam_norm
                state['trust_ratio'] = trust_ratio

                if N_sma >= 5:
                    p_data_fp32.addcdiv_(-radam_step_size * group['lr'] * trust_ratio, exp_avg, denom)
                else:
                    p_data_fp32.add_(-radam_step_size * group['lr'] * trust_ratio, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss
'''
'''
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
'''
'''
                print(i)
                if i[1]>=0.15:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                #if abs(i[0]-i[1])<1:
                #    y_pred.append(1)
                #else:
                #    y_pred.append(np.argmax(i))            
            #print(y_pred)
'''

Sensitivity = []
Specificity = []

Precisions = []
Recalls = []
prob = []
trues = []
#2. evaluate func
def evaluate(val_loader,model,criterion,T):
    #2.1 define meters
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    #2.2 switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    y_actu = []
    y_pred = []
    with torch.no_grad():
        for i,(input,target) in enumerate(val_loader):
            input = Variable(input).cuda()
            for i in np.array(target):
                y_actu.append(i)
                trues.append(i)
            #print(y_actu)
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            #target = Variable(target).cuda()
            #2.2.1 compute output
            output,_ = model(input)
            #smax = nn.Softmax(1)
            #smax_out = smax(output)
            m = nn.Sigmoid()
            smax_out = m(output)
            loss = criterion(output,target)
            #for i in np.array(smax_out.cpu()):
            #    prob.append(i[1])
            #    if i[1]>=T:
            #        y_pred.append(1)
            #        
            #    else:
            #        y_pred.append(0)

            for i in np.array(smax_out.cpu()):
                y_pred.append(np.argmax(i))
            #2.2.2 measure accuracy and record loss
            precision1,precision2 = accuracy(output,target,topk=(1,1))
            losses.update(loss.item(),input.size(0))
            top1.update(precision1[0],input.size(0))
            top2.update(precision2[0],input.size(0))
        
        print(len(trues))
        print(len(prob))
        '''
        C2= confusion_matrix(y_actu, y_pred, labels=[0, 1])
        tn,fp,fn,tp = C2.ravel()
        Sensitivity.append(tp/(tp+fn))
        Specificity.append(1-(fp/(fp+tn)))
        Precisions.append(tp/(tp+fp))
        Recalls.append(tp/(tp+fn))
        #print("\n")
        #print(tp/(tp+fn),1-(fp/(fp+tn)))
        #print("\n")
        print(y_actu)
        print(y_pred)
        '''
        cm = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)
        #print("\n")
        print(cm)
    return [losses.avg,top1.avg,top2.avg]

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
                y_pred = model(image_var)
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

    #define model
    #model_name = 'resnet50' # could be fbresnet152 or inceptionresnetv2
    #model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    #model = make_model('resnet18', num_classes=config.num_classes, pretrained=True)
    #model = torch.nn.DataParallel(model)
    #model =  models.resnet34(pretrained=False,num_classes=config.num_classes)
    #model = LambdaResNet18(num_classes=config.num_classes)
    
    #model = seresnet18(num_classs=config.num_classes)
    model = CTNet(num_classes=config.num_classes, num_queries=1, hidden_dim=512, nheads=8, num_encoder_layers=6, num_decoder_layers=6,branch_num=1) 
    #print(get_n_params(model))
    model = torch.nn.DataParallel(model)
    #print(model)
    #model = resnet10()
    #model = attention56()
    #model = seresnet34(num_classs=config.num_classes)
    #print(model.keys())
#    print(model.input_size)
#    print(model)
#    summary(model, torch.rand((1, 3, 512, 512)))
    #final_layer = nn.Conv2d(2048, config.num_classes, kernel_size=1, bias=True)
    #nn.init.xavier_uniform(final_layer.weight)
    #nn.init.constant(final_layer.bias, 0.1)
    #nn.init.kaiming_normal_(final_layer, mode='fan_out', nonlinearity='relu')
    #model.last_linear = final_layer
    #print(model_name)
    #print(model)
    #summary(model, torch.rand((1, 3, 512, 512)))
    #model2 = make_model('dpn92', num_classes=config.num_classes, pretrained=True)
    
    #if cuda_avail:
    model.cuda()

    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=config.weight_decay)
    
    #optimizer = Lookahead(optimizer)

    Loss = LabelSmoothingLoss(config.num_classes, smoothing=0.1).cuda()

    #lr_scheduler = CosineWarmupLr(optimizer, 320, 40,
    #                          base_lr=config.lr, warmup_epochs=1)
    #optimizer = optim.Adam(model.parameters(),lr = config.lr,amsgrad=True,weight_decay=config.weight_decay)
    #weights = torch.tensor([3.,4.2,5.,4.3])
    #criterion = nn.CrossEntropyLoss(weight=weights).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion = CircleLoss(m=0.25, gamma=30)
    #criterion = FocalLoss().cuda()
    log = Logger()
    log.open(config.logs + "log_train.txt",mode="a")
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    #4.3 some parameters for  K-fold and restart model
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False
    test_only = False
    eval_only = False #True  False


    #4.4 restart the training process
    if resume:
        checkpoint = torch.load(config.weights +config.model_name+'/' + str(fold) + "/checkpoint.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    #4.5 get files and split for K-fold dataset
    #4.5.1 read files
    train_ = get_files(config.train_data,"train")
    val_ = get_files(config.train_data,"val")
 
    
    #4.5.4 load dataset
    train_dataloader = DataLoader(CTDataset(train_,train=True),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=True,num_workers=4)
    #val_list for x ray
    val_dataloader = DataLoader(CTDataset(val_,val=True),batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,pin_memory=False,num_workers=4)
    #test_dataloader = DataLoader(ChaojieDataset(test_files,test=True),batch_size=config.batch_size*2,shuffle=False,pin_memory=False)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,"max",verbose=1,patience=3)
    scheduler =  optim.lr_scheduler.StepLR(optimizer,step_size = 50,gamma=0.1)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs, T_mult=2, eta_min=0.00001)
    #4.5.5.1 define metrics
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top2 = AverageMeter()
    valid_loss = [np.inf,0,0]
    model.train()
    #logs
    log.write('** start training here! **\n')
    log.write('                           |------------ VALID -------------|----------- TRAIN -------------|------Accuracy------|------------|\n')
    log.write('lr       iter     epoch    | loss   top-1  top-2            | loss   top-1  top-2           |    Current Best    | time       |\n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')
    #4.5.5 train
    start = timer()
    if eval_only:
        #best_model = torch.load(config.best_models +config.model_name+os.sep+ str(fold) +os.sep+ 'model_best.pth.tar')
        best_model = torch.load(config.weights +'x-ray/'+ 'lowest_loss.pth.tar')
        model.load_state_dict(best_model["state_dict"])
        #valid_loss = evaluate(val_dataloader,model,criterion,0.5)
        df = pd.DataFrame({'true':trues,'prob':prob})
        df.to_csv('gt.csv',index=False)
        
        for i in tqdm(range(201)):
            valid_loss = evaluate(val_dataloader,model,criterion,i*0.005)
        df = pd.DataFrame({'Sensitivity':Sensitivity,'Specificity':Specificity})
        df.to_csv('roc.csv',index=False)
        df = pd.DataFrame({'Precisions':Precisions,'Recalls':Recalls})
        df.to_csv('prc.csv',index=False)


        
        return

    if test_only:
        best_model = torch.load(config.best_models +config.model_name+os.sep+ str(fold) +os.sep+ 'model_best.pth.tar')
        model.load_state_dict(best_model["state_dict"])
        test(test_dataloader,model,fold)
    total_loss = 10
    
    for epoch in range(start_epoch,config.epochs):
        scheduler.step(epoch)
        # train
        #global iter
        for iter,(input,target) in enumerate(train_dataloader):
            #4.5.5 switch to continue train process
            model.train()
            input = Variable(input).cuda()
            #print(input.shape)
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            #target = Variable(target).cuda()
            output,_ = model(input)
            loss = criterion(output,target)
            #loss = Loss(output,target)

            precision1_train,precision2_train = accuracy(output,target,topk=(1,1))
            train_losses.update(loss.item(),input.size(0))
            train_top1.update(precision1_train[0],input.size(0))
            train_top2.update(precision2_train[0],input.size(0))
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            lr = get_learning_rate(optimizer)
            print('\r',end='',flush=True)
            print('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f         | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (\
                         lr, iter/len(train_dataloader) + epoch, epoch,
                         valid_loss[0], valid_loss[1], valid_loss[2],
                         train_losses.avg, train_top1.avg, train_top2.avg,str(best_precision_save),
                         time_to_str((timer() - start),'min'))
            , end='',flush=True)
        #evaluate
        lr = get_learning_rate(optimizer)
        #evaluate every half epoch
        valid_loss = evaluate(val_dataloader,model,criterion,0.5)#criterion Loss
        loss_min = False
        if valid_loss[0] < total_loss:
            total_loss = valid_loss[0]
            loss_min=True
        #valid_loss = [0.5,0.5,0.5]
        is_best = False
        is_best = valid_loss[1] >= best_precision1
        best_precision1 = max(valid_loss[1],best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass

        
        if is_best:
            save_checkpoint({
                        "epoch":epoch + 1,
                        "model_name":config.model_name,
                        "state_dict":model.state_dict(),
                        "best_precision1":best_precision1,
                        "optimizer":optimizer.state_dict(),
                        "fold":fold,
                        "valid_loss":valid_loss,
            },is_best,loss_min,fold)

        if loss_min:
            save_checkpoint({
                        "epoch":epoch + 1,
                        "model_name":config.model_name,
                        "state_dict":model.state_dict(),
                        "best_precision1":best_precision1,
                        "optimizer":optimizer.state_dict(),
                        "fold":fold,
                        "valid_loss":valid_loss,
            },is_best,loss_min,fold)
        #adjust learning rate
        #scheduler.step(valid_loss[1])
        print("\r",end="",flush=True)
        log.write('%0.4f %5.1f %6.1f        | %0.3f  %0.3f  %0.3f          | %0.3f  %0.3f  %0.3f         |         %s         | %s' % (\
                        lr, 0 + epoch, epoch,
                        valid_loss[0], valid_loss[1], valid_loss[2],
                        train_losses.avg,    train_top1.avg,    train_top2.avg, str(best_precision_save),
                        time_to_str((timer() - start),'min'))
                )
        log.write('\n')
        time.sleep(0.01)
    


if __name__ =="__main__":
    main()










































