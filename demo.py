import os
import pathlib
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
from torch.utils.data import DataLoader
from utils.core import evaluate
from utils.builder import *
from utils.utils import *
from utils.meter import AverageMeter
from utils.logger import Logger
from utils.module import MLPHead
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ResNet(nn.Module):
    def __init__(self, arch='resnet50', num_classes=200, pretrained=True, activation='relu', classifier='mlp-1'):
        super().__init__()
        assert arch in torchvision.models.__dict__.keys(), f'{arch} is not supported!'
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.feat_dim = resnet.fc.in_features
        self.neck = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if classifier == 'linear':
            self.classfier_head = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
            init_weights(self.classfier_head, init_method='He')
        elif classifier.startswith('mlp'):
            sf = float(classifier.split('-')[1])
            self.classfier_head = MLPHead(self.feat_dim, mlp_scale_factor=sf, projection_size=num_classes, init_method='He', activation='relu')
        else:
            raise AssertionError(f'{classifier} classifier is not supported.')
        self.proba_head = torch.nn.Sequential(
            MLPHead(self.feat_dim, mlp_scale_factor=1, projection_size=3, init_method='He', activation=activation),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N = x.size(0)
        x = self.backbone(x)
        x = self.neck(x).view(N, -1)
        logits = self.classfier_head(x)
        prob = self.proba_head(x)
        return {'logits': logits, 'prob': prob}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--nclasses', type=int, required=True)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    init_seeds()
    device = set_device(args.gpu)

    if args.dataset.startswith('web-'):
        transform = build_transform(rescale_size=448, crop_size=448)
        dataset = build_webfg_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
    else:
        transform = build_transform(rescale_size=256, crop_size=224)
        dataset = build_food101n_dataset(os.path.join('Datasets', args.dataset), transform['train'], transform['test'])
    net = ResNet(arch='resnet50', num_classes=args.nclasses).to(device)
    net.load_state_dict(torch.load(args.model_path))
    test_loader = DataLoader(dataset['test'], batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    test_accuracy = evaluate(test_loader, net, device)['accuracy']
    
    print(f'Test accuracy: {test_accuracy:.3f}')
