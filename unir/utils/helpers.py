import functools

import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


def print_networks(model):
    print('---------- Networks initialized -------------')
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()

    print('[{}] Total number of parameters : {:.3f} M'.format(model.__class__.__name__, num_params / 1e6))
    print('-----------------------------------------------')


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type is None:
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_net(net, init_type='normal', device='cuda:0', **kwargs):
    print('------------ {} ----------'.format(str(net.__class__.__name__)))
    init_weights(net, init_type, **kwargs)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters : {:.3f} M'.format(num_params / 1e6))
    print('-----------------------------------------------')
    return net.to(device)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'dirac':
                init.dirac_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Init type: {}, gain: {}'.format(init_type, gain))
    net.apply(init_func)


def get_scheduler(optimizer, lr_policy):
    if lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def safe_dict_get(dict, key, logger):
    try:
        dct = dict[key]
    except KeyError:
        logger.error("Bad option ! Choices are : {} ".format(" - ".join(dict.keys())))
        raise
    return dct
