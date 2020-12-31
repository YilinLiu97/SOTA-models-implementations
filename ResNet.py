import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_chn, out_chn, stride=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_chn, out_chn, stride=1, dilation=1):
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_chns, out_chns, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.downsample = downsample
        self.in_chns, self.out_chns = in_chns, out_chns

        stride = 2 if self.downsample is not None else 1
        self.conv1 = conv3x3(self.in_chns, self.out_chns, stride)
        self.bn1 = nn.BatchNorm2d(self.out_chns)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.out_chns, self.out_chns)
        self.bn2 = nn.BatchNorm2d(self.out_chns)                

    def forward(self, x):
        identity = x
       
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        print('out: ', out.shape)

        if self.downsample is not None:
           identity = self.downsample(x)
        print('identity: ', identity.shape)    

        out += identity
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    def __init__(self, in_chns, out_chns, block_type='Basic', layers=[3,4,6,3]):
        super(ResNet, self).__init__()
       
        self.inplanes = 64
        self.init_conv = nn.Conv2d(in_chns, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BasicBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#        self.fc = nn.Linear(512 * block.expansion, 1000)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
           downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
           )

        layers = [] 
        layers.append(block(self.inplanes, planes, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, None))

        return nn.Sequential(*layers)

    def forward(self, x): 
        x = self.init_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

 #       x = self.avgpool(x)
 #       x = torch.flatten(x, 1)
 #       x = self.fc(x)

        return x


'''
x = torch.rand(1,576,256,256)
model = ResNet(576,1)
out = model(x)
print(out.shape)
'''
