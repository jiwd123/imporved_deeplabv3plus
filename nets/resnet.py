import torch
import torch.nn as nn
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=6):
        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        fc = x4.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x4,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att, x1, x2, x3