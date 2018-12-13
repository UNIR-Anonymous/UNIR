import logging

from torch.utils.data import DataLoader

from unsup_it.dataset.celebA import CelebALoader
from unsup_it.dataset.lsun import LSUNLoader
from unsup_it.dataset.recipe import RecipeLoader
from unsup_it.module.corruption import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataset_default = {
    'common': {
        'batch_size': 16,
        'num_workers': 8,
    },
    'celebA': {
        'filename': '...',
        'nc': 3,
        'im_size': 64,
    },
    'LSUN': {
        'filename': '...',
        'nc': 3,
        'im_size': 64,
    },
    'recipe': {
        'filename': '...',
        'nc': 3,
        'im_size': 64,
    },
}

dataset_funcs = {
    'celebA': CelebALoader,
    'LSUN': LSUNLoader,
    'recipe': RecipeLoader,
}

corruption_config = {
    'common': {
    },
    'keep_patch': {
        'size_percent': (0.3, 1),
    },
    'remove_pix': {
        'percent': 0.9,
    },
    'remove_pix_dark': {
        'percent': 0.9,
    },
    'conv_noise': {
        'conv_size': 5,
        'noise_variance': 0.3,
    },
}
corruption_funcs = {
    "keep_patch": KeepPatch,
    "remove_pix": RemovePixel,
    "remove_pix_dark": RemovePixelDark,
    "conv_noise": ConvNoise,
}


def corruption(ex):
    @ex.config
    def config():
        corruption = {
            'name': 'remove_pix_dark'
        }
        corruption.update(corruption_config['common'])
        corruption.update(corruption_config[corruption['name']])

    @ex.capture
    def create(corruption, im_size):
        kwargs = corruption.copy()
        name = kwargs.pop('name')
        H = corruption_funcs[name](**kwargs, im_size=im_size)
        return H

    return create


def dataset(ex):
    @ex.config
    def config():
        dataset = {
            'name': 'celebA'
        }

        dataset.update(dataset_default['common'])
        dataset.update(dataset_default[dataset['name']])

    create_corruption = corruption(ex)

    @ex.capture
    def create(dataset):
        kwargs = dataset.copy()
        batch_size = kwargs.pop('batch_size')
        num_workers = kwargs.pop('num_workers')
        nc = kwargs.pop("nc")
        name = kwargs.pop('name')
        im_size = kwargs.pop('im_size')

        corruption = create_corruption(im_size=im_size)

        ds = dataset_funcs[name](is_train=True, measurement=corruption, **kwargs)
        test_ds = dataset_funcs[name](is_train=False, measurement=corruption, **kwargs)

        dl = DataLoader(dataset=ds,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        pin_memory=False,
                        num_workers=num_workers,
                        )

        test_dl = DataLoader(dataset=test_ds,
                             batch_size=dataset['batch_size'],
                             shuffle=False,
                             drop_last=True,
                             pin_memory=False,
                             num_workers=dataset['num_workers'],
                             )
        logger.info("loaded in {} : \n \t \t Train {} images (num batch {})  \n \t \t Test  {} images  (num batch {})"
                    .format(dataset["filename"], len(ds), len(dl), len(test_ds), len(test_dl)))

        return {
                   'train': dl,
                   'test': test_dl,
               }, corruption, nc

    return create
