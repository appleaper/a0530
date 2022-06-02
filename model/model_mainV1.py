import torch.nn as nn
from model.model_blockV1 import *
from torch.hub import load_state_dict_from_url
from model.model_blockV1 import model_urls
import torch
import os
from model.mobilenet import MobileNetV2
from model.vgg import VGG, make_layers, cfgs

def Resnet18(conf):
    model = resnet18()
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, conf.num_class)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['resnet18'], progress=True)
            model.load_state_dict(state_dict)
    return model

def Resnet50(conf):
    model = resnet50()
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['resnet50'], progress=True)
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, conf.num_class)
    return model

def Resnet101(conf):
    model = resnet101()
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['resnet101'], progress=True)
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, conf.num_class)
    return model

def Resnext101_32x8d(conf):
    model = resnext101_32x8d()
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['resnext101_32x8d'], progress=True)
            model.load_state_dict(state_dict)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, conf.num_class)
    return model

def mobilenet_v2(conf):
    '''mobilnetV2'''
    model = MobileNetV2()
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['mobilenet'], progress=True)
            model.load_state_dict(state_dict)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, conf.num_class),
    )
    return model

def vgg16(conf):
    '''VGG模型'''
    model = VGG(make_layers(cfgs['D']))
    if os.path.exists(conf.pre_train_path):
        state_dict = torch.load(conf.pre_train_path, map_location=conf.device)
        model.load_state_dict(state_dict)
    else:
        if conf.download:
            state_dict = load_state_dict_from_url(model_urls['mobilenet'], progress=True)
            model.load_state_dict(state_dict)

    model.classifier =  nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, conf.num_class),
    )
    return model