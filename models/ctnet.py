"""

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor

class NormLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(NormLinear, self).__init__(in_features, out_features, bias=False)

    def forward(self, inp: Tensor) -> Tensor:
        return nn.functional.linear(nn.functional.normalize(inp),
                                    nn.functional.normalize(self.weight))

class BasicResidualSEBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels * self.expansion, 3, padding=1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)

        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class BottleneckResidualSEBlock(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride, r=16):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, 1),
            nn.BatchNorm2d(out_channels * self.expansion),
            nn.ReLU(inplace=True)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels * self.expansion, out_channels * self.expansion // r),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * self.expansion // r, out_channels * self.expansion),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        shortcut = self.shortcut(x)

        residual = self.residual(x)
        squeeze = self.squeeze(residual)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)

        x = residual * excitation.expand_as(residual) + shortcut

        return F.relu(x)

class SEResNet(nn.Module):

    def __init__(self, block, block_num, class_num=2):
        super().__init__()

        self.in_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.stage1 = self._make_stage(block, block_num[0], 64, 1)
        self.stage2 = self._make_stage(block, block_num[1], 128, 2)
        self.stage3 = self._make_stage(block, block_num[2], 256, 2)
        self.stage4 = self._make_stage(block, block_num[3], 516, 2)

        #self.linear = nn.Linear(self.in_channels, class_num)

        self.linear = NormLinear(self.in_channels, class_num)
    
    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        feat = x

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        return x, feat

    
    def _make_stage(self, block, num, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        while num - 1:
            layers.append(block(self.in_channels, out_channels, 1))
            num -= 1
        
        return nn.Sequential(*layers)
        
def seresnet18(num_classs=2):
    return SEResNet(BasicResidualSEBlock, [2, 2, 2, 2], class_num = num_classs)

def seresnet34(num_classs=2):
    return SEResNet(BasicResidualSEBlock, [3, 4, 6, 3], class_num = num_classs)

def seresnet50(num_classs=2):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 6, 3], class_num = num_classs)

def seresnet101(num_classs=2):
    return SEResNet(BottleneckResidualSEBlock, [3, 4, 23, 3], class_num = num_classs)

def seresnet152(num_classs=2):
    return SEResNet(BottleneckResidualSEBlock, [3, 8, 36, 3], class_num = num_classs)




class CTNet(nn.Module):
    def __init__(self, num_classes, num_queries, hidden_dim, nheads,
                 num_encoder_layers, num_decoder_layers,branch_num): #fot  
        super().__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        #We take convolutional layers from ResNet-18 model
        self.backbone = nn.Sequential(*list(seresnet18(num_classs=2).children())[:-1])
        self.conv = nn.Conv2d(516, hidden_dim, 1)
        #self.avgpool = nn.AvgPool2d(7)# multi-cell
        #self.avgpool = nn.AvgPool2d(7)# single-cell
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.branch_num = branch_num
        
        encoder_layers = TransformerEncoderLayer(hidden_dim, nheads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.linear_class = nn.Linear(hidden_dim*4*branch_num, num_classes) 

    def forward(self, inputs):
        
        x = self.backbone(inputs)
        feat = x
        h = self.conv(x)
       
        p = self.avgpool(h)
        if self.branch_num==1:
            p = p.view(p.size(0), -1)
            pos = p
        else: 
            e = torch.reshape(p, (p.shape[0], p.shape[2]*p.shape[3], p.shape[1]))
            e = self.transformer_encoder(e)
            e = e.view(e.size(0), -1)
            p = p.view(p.size(0), -1)
            pos = torch.cat((e,p),1)

        return self.linear_class(pos),feat#, self.linear_bbox(e).sigmoid()
        

#256 for 224
ctnet = CTNet(num_classes=2, num_queries=1, hidden_dim=512, nheads=8, num_encoder_layers=6, num_decoder_layers=6,branch_num=1) 

mynet = seresnet18(num_classs=2)

input = torch.rand([2,32, 224,224], dtype=torch.float32)

output,_ = mynet(input)

print(output.shape)

logits,_ = ctnet(input)

print(logits.shape)
