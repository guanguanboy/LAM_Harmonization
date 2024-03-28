import os
import torch
from collections import OrderedDict
import torch.nn as nn
MODEL_DIR = './ModelZoo/models'


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'SPANET'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'SPANET': {
        'Base': 'latest_net_G_0322.pth',
    },
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)
        elif model_name == 'SPANET':
            from .NN.SPANET_arch import SPANet
            input_size = 256
            depths=[1, 1, 1, 1, 2, 1, 1, 1,1]
            embed_dim = 32

            net = SPANet(img_size=input_size, in_chans=3, dd_in=4, embed_dim=embed_dim,depths=depths, win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False)

            #net = nn.DataParallel(net)

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        #if k[:7] == "module.":
            #k = k[7:]
        if k[:7] == "spanet.":
            k = k[7:]
        new_state_dict[k] = v
    net.load_state_dict(new_state_dict, strict=False)
    
    #net.load_state_dict(state_dict)

    return net




