from unsup_it.closures import *
from unsup_it.utils.helpers import *

closure_default_config = {
    'common': {
    },
    'unsup_ir': {
        'backloss_measurements': 2.,
    },
    'paired_variant': {
        'lambda_reconstruction': 2.,
    },
    'unpaired_variant': {
        'backloss_measurements': 2.,
    },
}

closure_funcs = {
    'unsup_ir': UnsupervisedImageReconstruction,
    'paired_variant': PairedVariant,
    'unpaired_variant': UnpairedVariant,
}


def closure(ex):
    logger = ex.logger

    @ex.config
    def config(_log):
        closure = {
            'name': 'unsup_ir',
        }
        closure.update(closure_default_config['common'])
        closure.update(safe_dict_get(closure_default_config, closure['name'], _log))

    @ex.capture
    def create(mods, optims, device, measurement, closure, scheduler, _log):
        kwargs = closure.copy()
        name = kwargs.pop('name')
        return safe_dict_get(closure_funcs, name, _log)(mods, optims,
                                                        device=device,
                                                        measurement=measurement,
                                                        dict_scheduler=scheduler,
                                                        **kwargs)

    return create
