import torch
import torch.nn as nn
import torch.nn.functional as f


def separate_bn_prelu_params(model, ignored_params=[]):
    bn_prelu_params = []
    for m in model.modules():
        print(module.__class__)
        if isinstance(m, nn.BatchNorm2d):
            print('is BatchNorm2d')
            ignored_params += list(map(id, m.parameters()))  
            bn_prelu_params += m.parameters()
        if isinstance(m, nn.BatchNorm1d):
            print('is BatchNorm1d')
            ignored_params += list(map(id, m.parameters()))  
            bn_prelu_params += m.parameters()
        if isinstance(m, nn.PReLU):
            print('is PReLU')
            ignored_params += list(map(id, m.parameters()))
            bn_prelu_params += m.parameters()
    base_params = list(filter(lambda p: id(p) not in ignored_params, model.parameters()))
    return base_params, bn_prelu_params, ignored_params