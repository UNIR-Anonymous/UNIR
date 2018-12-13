import torch

default_lr_scheduler_config = {
    'none': {
        'lr_lambda': lambda epoch: 1,
    },
    'lambda': {
        'lr_lambda': lambda epoch: (1 - 1e-2) ** epoch,
    },
    'exponential': {
        'gamma': 0.995
    },
    'multi_step': {
        'milestones': [1, 2, 5, 10],
        'gamma': 0.1,
    },
    'reduce_on_plateau': {
        'mode': 'min',
        'factor': 0.2,
        'threshold': 0.01,
        'patience': 5,
    },
}

lr_scheduler_modules = {
    'none': torch.optim.lr_scheduler.LambdaLR,
    'lambda': torch.optim.lr_scheduler.LambdaLR,
    'step': torch.optim.lr_scheduler.StepLR,
    'multi_step': torch.optim.lr_scheduler.MultiStepLR,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}


def lr_scheduler(ex):
    @ex.config
    def config():
        lr_scheduler = {
            'name': 'exponential',
        }

        lr_scheduler.update(default_lr_scheduler_config[lr_scheduler['name']])

    @ex.capture
    def create(dict_optim, lr_scheduler):
        kwargs = lr_scheduler.copy()
        name = kwargs.pop('name')
        return {k: lr_scheduler_modules[name](v, **kwargs)
                for k, v in dict_optim.items()}

    return create
