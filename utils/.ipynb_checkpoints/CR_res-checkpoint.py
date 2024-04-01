import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
from torchvision import models
#######
import cv2
import matplotlib.pyplot as plt
#######

class Resnet152(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Resnet152, self).__init__()
        res_pretrained_features = models.resnet152(pretrained=True)
        self.slice1 = torch.nn.Sequential(*list(res_pretrained_features.children())[:-5])
        self.slice2 = torch.nn.Sequential(*list(res_pretrained_features.children())[-5:-4])
        self.slice3 = torch.nn.Sequential(*list(res_pretrained_features.children())[-4:-3])
        self.slice4 = torch.nn.Sequential(*list(res_pretrained_features.children())[-3:-2])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
       # h_relu4 = self.slice4(h_relu3)
        return [h_relu1, h_relu2, h_relu3]



class ContrastLoss_res(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss_res, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        # self.l1 = nn.SmoothL1Loss(reduction='mean')
        #self.weights = [ 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weights = [1.0 ,1.0 / 4, 1.0 / 8]
        self.ab = ablation

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            a, p, n = a_vgg[i], p_vgg[i], n_vgg[i]
            d_ap = self.l1(a, p.detach())
            if not self.ab:
                d_an = self.l1(a, n.detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


'''
class ContrastLoss_res1(nn.Module):
    # 是一个用于计算对比损失（Contrastive Loss）的PyTorch模型类，通常用于学习如何在给定锚点（anchor）和正例（positive）样本之间增加相似度，
    # 同时在给定锚点和负例（negative）样本之间减少相似度
    def __init__(self, ablation=False):
        super(ContrastLoss_res1, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction='mean')
        self.ab = ablation

    def forward(self, a, p):
        # 将a的rgb三通道分离出来
        loss=0
        loss=self.l1(a,p.detach())
        # loss = self.l1(a[:,0:1,:,:], p[:,0:1,:,:].detach()) + self.l1(a[:,1:2,:,:], p[:,1:2,:,:].detach()) + self.l1(a[:,2:3,:,:], p[:,2:3,:,:].detach())
        return loss

        
        B1, G1, R1 = a[:,0:1,:,:],a[:,1:2,:,:],a[:,2:3,:,:]
        B2, G2, R2 = p[:,0:1,:,:],p[:,1:2,:,:],p[:,2:3,:,:]
        loss = self.l1(B1, B2.detach()) + self.l1(G1, G2.detach()) + self.l1(R1, R2.detach())
        


class ContrastLoss_res2(nn.Module):
    # 是一个用于计算对比损失（Contrastive Loss）的PyTorch模型类，通常用于学习如何在给定锚点（anchor）和正例（positive）样本之间增加相似度，
    # 同时在给定锚点和负例（negative）样本之间减少相似度
    def __init__(self, ablation=False):

        super(ContrastLoss_res2, self).__init__()
        self.vgg = Resnet152().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [ 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation

    def forward(self, a, p, n):

        loss = 0
        d_ap = self.l1(a, p.detach())
        if not self.ab:
            d_an = self.l1(a, n.detach())
            contrastive = d_ap / (d_an + 1e-7)
        else:
            contrastive = d_ap

        loss +=  contrastive
        return loss
'''