import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']





def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)

                     


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #print(out.size())
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classeses=1000, num_neurons=128):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=1, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool1d(2)
        #self.fc = nn.Linear(512 * block.expansion, num_classeses)
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion*block.expansion, num_neurons * block.expansion),#####2048
        nn.BatchNorm1d(num_neurons * block.expansion),
        nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(num_neurons * block.expansion, num_classeses)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc1(x)
        ca = self.fc2(y)

        return x, y, ca


def resnet50(args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classeses=args.num_classes, num_neurons=args.num_neurons)
    

    return model


def resnet101(args, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classeses=args.num_classes, num_neurons=args.num_neurons)
    if args.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet152(args, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classeses=args.num_classes, num_neurons=args.num_neurons)
    if args.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet152'])
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
    

def resnet(args, **kwargs):  # Only the ResNet34 is supported.
    print("==> creating model '{}' ".format(args.arch))
    if args.arch == 'resnet50':
        return resnet50(args)
    elif args.arch == 'resnet101':
        return resnet101(args)
    elif args.arch == 'resnet152':
        return resnet152(args)
    else:
        raise ValueError('Unrecognized model architecture', args.arch)
