import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import  torchvision.transforms as transforms
from PIL import ImageStat

import torch
from mmengine.model import constant_init, kaiming_init
from torch.nn import init as init

from ops_dcnv3.modules.dcnv3 import DCNv3_pytorch


class Mix(nn.Module):
    def __init__(self,m=-0.80):
        super(Mix, self).__init__()
        w=torch.nn.Parameter(torch.FloatTensor([m]),requires_grad=True)
        w=torch.nn.Parameter(w,requires_grad=True)
        self.w=w
        self.mix_block=nn.Sigmoid()

    def forward(self,fea1,fea2):
        mix_factor=self.mix_block(self.w)
        out=fea1*mix_factor.expand_as(fea1)+fea2*(1-mix_factor.expand_as(fea2))
        return out

class Mix2(nn.Module):
    def __init__(self,m=-0.80):
        super(Mix2, self).__init__()
        w=torch.nn.Parameter(torch.FloatTensor([m]),requires_grad=True)
        w=torch.nn.Parameter(w,requires_grad=True)
        self.w=w
        self.mix_block=nn.ReLU()

    def forward(self,x):
        mix_factor=self.mix_block(self.w)
        out=x*mix_factor

        return out


class MSFM(nn.Module):
    def __init__(self, features, r=2, L=24) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.convll = nn.Sequential(
            nn.Conv2d(features, features , kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )
        self.convl = nn.Sequential(
            nn.Conv2d(features, features , kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )
        self.convm = nn.Sequential(
            nn.Conv2d(features, features , kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.GELU()
        )


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.Sequential(
            nn.Conv2d(features, features, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(features, features, 1, 1, 0)
        )
        self.fcs2 = nn.Sequential(
            nn.Conv2d(features, features, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(features, features, 1, 1, 0)
        )

        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.pa = nn.Sequential(
            nn.Conv2d(features, features // 8, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(features // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sigmod=nn.Sigmoid()
        self.cov7=nn.Conv2d(2,1,kernel_size=1,bias=True)

        self.mix=Mix2(m=0.8)
        self.out = nn.Sequential(
            nn.Conv2d(features*2, features , 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.norm = nn.BatchNorm2d(features)
        self.norm1 = nn.BatchNorm2d(features)

    def forward(self, x):
        lowlow = self.convll(x)
        low = self.convl(lowlow)
        middle = self.convm(low)

        emerge1 = low + lowlow+middle

        emerge2 = self.gap(emerge1)
        emerge2 = self.softmax(self.fcs(emerge2))

        fea_high = emerge2 * emerge1


        max_out,_=torch.max(emerge1,dim=1,keepdim=True)
        avg_out=torch.mean(emerge1,dim=1,keepdim=True)
        spa_out=self.sigmod(self.cov7(torch.cat([max_out,avg_out],dim=1)))

        fea_high2= spa_out*emerge1

        out = self.out(torch.cat([fea_high , fea_high2],dim=1 ))

        return out + self.mix(x)


class MSPLCK(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm3 = nn.BatchNorm2d(dim)

        self.conv1=nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

        self.conv3_19 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, groups=dim, dilation=3, padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim, dim, kernel_size=5, padding=6, groups=dim, dilation=3, padding_mode='reflect')
        # self.conv3_7 = DCNv3_pytorch()
        self.conv3_7 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, groups=dim, dilation=3, padding_mode='reflect')
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim*3, dim *4, 1),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

        self.sf=MSFM(dim)


    def forward(self, x):

        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_19(x), self.conv3_13(x), self.conv3_7(x)], dim=1)
        x = self.mlp2(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.conv3(x)
        x=self.sf(x)
        x = x + identity

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [MSPLCK(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out




class TSNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 96, 48, 24],
                 depths=[1, 1, 2, 1, 1]):
        super(TSNet, self).__init__()

        # setting
        self.patch_size = 4

        # DCN
        self.dcn1 = DCNv3_pytorch(kernel_size=3,group=4,channels=96,center_feature_scale=True)
        self.gelu=nn.GELU()


        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])

        self.patch_merge2 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2], kernel_size=3)

        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])

        assert embed_dims[1] == embed_dims[3]
        self.fusion1 = SKFusion(embed_dims[3])

        self.layer4 = BasicLayer(dim=embed_dims[3], depth=depths[3])

        self.patch_split2 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])

        assert embed_dims[0] == embed_dims[4]
        self.fusion2 = SKFusion(embed_dims[4])

        self.layer5 = BasicLayer(dim=embed_dims[4], depth=depths[4])

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(96, 96, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(96, 96, 1, padding=0, bias=True),
            nn.Softmax(dim=1)
        )

        self.mix1=Mix2(m=0.6)
        self.mix2=Mix2(m=0.4)


    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)
        skip2 = x

        x = self.patch_merge2(x)
        x = self.layer3(x)

        # ALM
        skip3=x
        x = self.dcn1(x)
        x=self.ca(x)*x+self.mix1(skip3)

        skip3 = x
        x = self.dcn1(x)
        x = self.ca(x) * x + self.mix1(skip3)


        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip2(skip2)]) + x
        x = self.layer4(x)
        x = self.patch_split2(x)

        x = self.fusion2([x, self.skip1(skip1)]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x


    def forward(self, x):
        H, W = x.shape[2:]

        x = self.check_image_size(x)


        # feat=[1,4,256,256]
        feat = self.forward_features(x)
        # K=[:,1,256,256],B=[:,3,256,256]
        K, B = torch.split(feat, (1, 3), dim=1)

        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


class TSNet2(nn.Module):
    def __init__(self, in_chans=3, out_chans=4,
                 embed_dims=[24, 48, 24],
                 depths=[1, 2, 1]):
        super(TSNet2, self).__init__()

        # setting
        self.patch_size = 4

        # DCN
        self.dcn1 = DCNv3_pytorch(kernel_size=3,group=4,channels=48,center_feature_scale=True)
        self.gelu=nn.GELU()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.layer1 = BasicLayer(dim=embed_dims[0], depth=depths[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1], kernel_size=3)

        self.layer2 = BasicLayer(dim=embed_dims[1], depth=depths[1])


        self.patch_split1 = PatchUnEmbed(
            patch_size=2, out_chans=embed_dims[2], embed_dim=embed_dims[1])

        self.fusion1 = SKFusion(embed_dims[0])

        self.layer3 = BasicLayer(dim=embed_dims[2], depth=depths[2])

        self.patch_unembed = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=embed_dims[2], kernel_size=3)

        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)

        assert embed_dims[0] == embed_dims[2]


        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 48, 1, padding=0, bias=True),
            nn.GELU(),
            # nn.ReLU(True),
            nn.Conv2d(48, 48, 1, padding=0, bias=True),
            nn.Softmax(dim=1)
        )
        self.mix1=Mix2(m=0.6)
        self.mix2=Mix2(m=0.4)
    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layer1(x)
        skip1 = x

        x = self.patch_merge1(x)
        x = self.layer2(x)

        # ALM
        skip3 = x
        x = self.dcn1(x)
        x = self.ca(x) * x + self.mix1(skip3)

        skip3 = x
        x = self.dcn1(x)
        x = self.ca(x) * x + self.mix1(skip3)

        x = self.patch_split1(x)

        x = self.fusion1([x, self.skip1(skip1)]) + x
        x = self.layer3(x)
        x = self.patch_unembed(x)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        # feat=[1,4,256,256]
        feat = self.forward_features(x)

        # K=[:,1,256,256],B=[:,3,256,256]
        K, B = torch.split(feat, (1, 3), dim=1)
        #print(K.shape,B.shape)
        #x=feat+x
        x = K * x - B + x
        x = x[:, :, :H, :W]
        return x


def TSNet_t():
    return TSNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[1, 1, 2, 1, 1])

def TSNet_t2():
    return TSNet2(
        embed_dims=[24, 48, 24],
        depths=[1, 2, 1])

def TSNet_s():
    return TSNet(
        embed_dims=[24, 48, 96, 48, 24],
        depths=[2, 2, 4, 2, 2])




