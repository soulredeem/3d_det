from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, BinRotLoss, alpha_prediction_loss, GlobalPoolL1Loss
# from models.losses import Position_loss
from models.decode import car_pose_decode, multi_pose_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import car_pose_post_process, multi_pose_post_process
from .base_trainer import BaseTrainer


class CarPoseLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CarPoseLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = GlobalPoolL1Loss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        # self.crit_rot = BinRotLoss()
        self.seg = alpha_prediction_loss
        self.opt = opt
        # self.position_loss = Position_loss(opt)

    def forward(self, outputs, batch, phase=None):
        opt = self.opt
        hp_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0
        seg_loss = 0
        output = outputs[0]

        if opt.hm_hp and not opt.mse_loss:
            output['hm_hp'] = _sigmoid(output['hm_hp'])

        hp_loss = self.crit_kp(output['hps'], batch['hps'], batch['hp_mask'])

        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss = self.crit_reg(output['hp_offset'], batch['hp_mask'], batch['hp_ind'], batch['hp_offset'])
        if opt.hm_hp and opt.hm_hp_weight > 0:
            hm_hp_loss = self.crit_hm_hp(output['hm_hp'], batch['hm_hp'])
        
        loss_stats = {'hp_loss': hp_loss, 'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss}
        return loss_stats, loss_stats


class CarPoseTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CarPoseTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):

        loss_states = ['hp_loss', 'hm_hp_loss', 'hp_offset_loss',  ]
        loss = CarPoseLoss(opt)
        return loss_states, loss

    

    def save_result(self, output, batch, results):

        hm_hp = output['hm_hp'] if self.opt.hm_hp else None
        hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
        dets = multi_pose_decode(output['hm'], output['wh'], output['hps'],
                                 reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])

        dets_out = multi_pose_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
