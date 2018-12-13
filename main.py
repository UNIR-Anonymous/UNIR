import logging
import os
import shutil
import time
from datetime import datetime

import torch
import torchvision.utils as vutils
from sacred import Experiment
from sacred.observers import FileStorageObserver

import unir.factory.closures as closure_factory
import unir.factory.dataset as dataset_factory
import unir.factory.lr_scheduler as lr_scheduler_factory
import unir.factory.modules as module_factory
import utils.external_resources as external
from utils.meters import AverageMeter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

ex = Experiment('unsup')
time_str = datetime.now().strftime("%a-%b-%d-%H:%M:%S")
exp_dir = os.path.join("...", ex.path, time_str)
os.makedirs(exp_dir)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    path = os.path.join(exp_dir, filename)
    try:
        torch.save(state, path)
        logging.info('saving to {}...'.format(path))
        if is_best:
            shutil.copyfile(path, os.path.join(exp_dir, 'model_best.pth.tar'))
    except:
        logging.warning('saving to {} FAILED'.format(path))


@ex.config
def config():
    device = "cuda:0"
    nepochs = 800
    exp_dir = exp_dir
    debug = False
    use_mongo = True
    if use_mongo and not debug:
        ex.observers.append(external.get_mongo_obs())
    else:
        ex.observers.append(FileStorageObserver.create(exp_dir))
    if debug:
        nepochs = 10

    niter = {
        'train': 400,
        'test': 20,
    }

    create_datasets = dataset_factory.dataset(ex)
    create_modules = module_factory.UnsupIR(ex)
    create_closure = closure_factory.closure(ex)
    create_scheduler = lr_scheduler_factory.lr_scheduler(ex)


@ex.automain
def main(_run, nepochs, niter, device, _config, create_datasets, create_modules, create_closure, create_scheduler,
         exp_dir):
    shutil.make_archive('./unir', 'zip', './unir')
    ex.add_resource("./unir.zip")

    logger.info('### Dataset ###')

    dsets, corruption, nc = create_datasets()
    logger.info('### Model and Optim ###')

    mods, optims = create_modules(nc=nc, corruption=_config["corruption"], closure_name=_config['closure']['name'])
    dict_scheduler = create_scheduler(dict_optim=optims)
    closure = create_closure(mods, optims, device, measurement=corruption, scheduler=dict_scheduler)

    logger.info('### Begin Training ###')
    best_mse = float('inf')
    for epoch in range(1, nepochs + 1):

        logger.info('### Starting epoch n°{} '.format(epoch))
        for split, dl in dsets.items():

            iter = 0
            with torch.set_grad_enabled(split == 'train'):
                batch_time = AverageMeter()
                data_time = AverageMeter()
                end = time.time()
                for batch in dl:
                    print_freq = niter[split] / 5
                    for var_name, var in batch.items():
                        batch[var_name] = var.to(device)
                    data_time.update(time.time() - end)
                    closure.forward(batch)
                    if split == 'train':
                        closure.backward()
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if iter % print_freq == 0:
                        logger.info('Epoch: [{0}] {split} [{1}/{2}]\t'
                                    'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                            epoch, iter, niter[split], split=split, batch_time=batch_time, data_time=data_time,
                        ))
                    iter += 1

                    if iter > niter[split]:
                        break

            meters, images = closure.step()
            ims = torch.cat([v for k, v in images.items() if v.dim() == 4 and v.shape[1] in [1, 3]])
            path = exp_dir + '/' + split + '_' + str(epoch) + '.png'
            try:
                vutils.save_image(ims, path, scale_each=True, normalize=True, nrow=dl.batch_size)
                logger.info("saving images in {path}".format(path=path))
            except:
                logger.warning("SAVING IMAGES FAILED")

                pass
            string_to_print = '*** '
            for name, v in meters.items():
                tag = 'meters' + '/' + name + '/' + split
                ex.log_scalar(tag, v, epoch)
                string_to_print += '** {name:10} {meters:.5f} '.format(split=split, name=name, meters=v)
            logger.info(string_to_print)

            if split == 'test':
                is_best = meters['loss_MSE'] < best_mse
                best_mse = max(meters['loss_MSE'], best_mse)
                _run.result = best_mse
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': mods['gen'].state_dict(),
                    'best_MSE': best_mse,
                }, is_best)
