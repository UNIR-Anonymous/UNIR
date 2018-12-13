import torch.nn as nn

import unsup_it.module.losses as losses
from .closure import *


class PairedVariant(Closure):
    def __init__(self, mods: dict, optims: dict, device: str, lambda_reconstruction: float = 0., measurement=None,
                 dict_scheduler=None):
        assert (measurement is not None)
        self.measurement = measurement
        self.netG = mods['gen']
        self.netD = mods['dis']
        self.netG_optim = optims['gen']
        self.netD_optim = optims['dis']
        self.lambda_reconstruction = lambda_reconstruction
        self.likelihood_loss = nn.MSELoss()
        self.prior_loss = losses.GANLoss().to(device)
        self.device = device
        self.dict_scheduler = dict_scheduler
        self.lr_gen = self.dict_scheduler['gen'].get_lr()[0]
        self.lr_dis = self.dict_scheduler['dis'].get_lr()[0]
        self._register_metrics(['loss_*', 'lr_*'])

    def _forward(self, input: dict):
        self.sample = input["sample"]
        self.theta = input["theta"]
        self.measured_sample = input['measured_sample']
        self.fake_sample = self.netG(self.measured_sample)
        return input

    def _backward_G(self):
        set_requires_grad(self.netD, False)
        self.netG_optim.zero_grad()

        fake_pair = torch.cat([self.measured_sample, self.fake_sample], 1)
        pred_fake = self.netD(fake_pair)
        self.loss_G = self.prior_loss(pred_fake, True)

        if self.lambda_reconstruction > 0:
            self.loss_MSE = self.likelihood_loss(self.fake_sample, self.sample.detach())
            self.loss_G = self.loss_G + self.loss_MSE * self.lambda_reconstruction

        self.loss_G.backward()
        self.netG_optim.step()
        return self.loss_G

    def _step(self):
        self.lr_gen = self.dict_scheduler['gen'].get_lr()[0]
        self.lr_dis = self.dict_scheduler['dis'].get_lr()[0]
        self.dict_scheduler['gen'].step()
        self.dict_scheduler['dis'].step()

    def _backward_D(self):
        set_requires_grad(self.netD, True)
        self.netD_optim.zero_grad()

        real_pair = torch.cat((self.measured_sample, self.sample), 1)
        pred_real = self.netD(real_pair)

        fake_pair = torch.cat((self.measured_sample, self.fake_sample), 1)
        pred_fake = self.netD(fake_pair.detach())

        # Real
        loss_D_real = self.prior_loss(pred_real, True)
        # Fake
        loss_D_fake = self.prior_loss(pred_fake, False)
        # Combined loss
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        self.loss_D.backward()
        self.netD_optim.step()

        return self.loss_D

    def _backward(self):
        self._backward_G()
        self._backward_D()
