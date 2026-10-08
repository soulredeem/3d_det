import torch
import torch.nn as nn
from .mobilenetv2 import *
import torch.utils.model_zoo as model_zoo

pretrain_path = '/opt/tiger/model_best.pth.tar'
def load_network(name):

    state_dict = torch.load(name)['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict

class MobileNet(MobileNetV2):
    'MobileNetV2: Inverted Residuals and Linear Bottlenecks - https://arxiv.org/abs/1801.04381'

    def __init__(self, outputs=[17], url=None):
        self.stride = 128
        self.url = url
        super().__init__(width_mult=0.35)
        self.outputs = outputs
        self.classifier = None

    def initialize(self):
        print('done!')
        #if self.url:
        #    self.load_state_dict(model_zoo.load_url(self.url),  strict=False)
        #print(torch.load(pretrain_path))
        self.load_state_dict(load_network(pretrain_path), strict=False)

    def forward(self, x):
        outputs = []
        for indx, feat in enumerate(self.features):
            x = feat(x)
            if indx in self.outputs:
                outputs.append(x)
        return outputs
        # return x
