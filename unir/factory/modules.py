import logging
import math

import torch

import unsup_it.utils.helpers as helpers
from unsup_it.module.SAGAN import ResNetGenerator, ResNetDiscriminator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generator_default_config = {
    'common': {
        'ngf': 32,
        'init': 'orthogonal',
        'optim': {
            'name': 'adam',
            'betas': (0.0, 0.999),
            'lr': 0.0001,
        },
    },
    'sagan_resnet': {
        "self_attention": True,
    },
}

discriminator_default_config = {
    'common': {
        'ndf': 32,
        'init': 'orthogonal',
        'optim': {
            'name': 'adam',
            'betas': (0.0, 0.999),
            'lr': 0.0004,
        },
    },
    'resnet': {

    }
}

gen_funcs = {
    'sagan_resnet': ResNetGenerator,
}

dis_funcs = {
    'resnet': ResNetDiscriminator,
}

optim_funcs = {
    'adam': torch.optim.Adam,
}


def create_optimizer(opt, params):
    opt = opt.copy()
    return optim_funcs[opt.pop('name')](params, **opt)


def generator(ex):
    @ex.config
    def config(modules):
        modules['gen'] = {
            'name': 'sagan_resnet'
        }
        modules['gen'].update(generator_default_config['common'])
        modules['gen'].update(generator_default_config[modules['gen']['name']])

    @ex.capture
    def create(modules, device, nc, corruption):
        kwargs = modules['gen'].copy()
        kwargs.update({'input_nc': nc, 'output_nc': nc})
        name = kwargs.pop('name')
        init_type = kwargs.pop('init')
        add_input = corruption['name'] in ['keep_patch', 'conv_noise']
        init_gain = 0.1 if add_input else math.sqrt(2)
        optim = kwargs.pop('optim')
        generator = gen_funcs[name](**kwargs)
        optimizer = create_optimizer(optim, generator.parameters())
        return helpers.init_net(generator, init_type, device, gain=init_gain), optimizer

    return create


def discriminator(ex):
    @ex.config
    def config(modules):
        modules['dis'] = {'name': 'resnet'}
        modules['dis'].update(discriminator_default_config['common'])
        modules['dis'].update(discriminator_default_config[modules['dis']['name']])

    @ex.capture
    def create(modules, device, nc):
        kwargs = modules['dis'].copy()
        kwargs.update({'input_nc': nc})
        name = kwargs.pop('name')
        init_type = kwargs.pop('init')
        init_gain = math.sqrt(2)
        optim = kwargs.pop('optim')
        dis = dis_funcs[name](**kwargs)
        optimizer = create_optimizer(optim, dis.parameters())
        return helpers.init_net(dis, init_type, device, gain=init_gain), optimizer

    return create


def UnsupIR(ex):
    @ex.config
    def config():
        modules = {}

    create_generator = generator(ex)
    create_discriminator = discriminator(ex)

    @ex.capture
    def create(device, nc, corruption, closure_name):
        # As we concatenate the sample for the pairde variant, we have to double the input channel
        # of the discriminator
        closure_mult = 2 if closure_name == 'paired_variant' else 1

        gen, optim_gen = create_generator(device=device, nc=nc, corruption=corruption)
        dis, optim_dis = create_discriminator(device=device, nc=nc * closure_mult)

        modules = {
            'gen': gen,
            'dis': dis,
        }

        optimizers = {
            'gen': optim_gen,
            'dis': optim_dis,
        }

        return modules, optimizers

    return create
