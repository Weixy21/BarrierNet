import math
import torch.nn as nn
from cvxopt import solvers, matrix

def build_cnn(filters, dropout=0., with_bn=True, with_gn=False,
              no_act_last_layer=False, activation='relu'):
    modules = nn.ModuleList()
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'elu':
        activation = nn.ELU()
    else:
        raise NotImplementedError(f'Not supported activation function {activation}')
    if with_gn: # mutually exclusive of gn and bn; bn default to True and will be overwritten by gn
        with_bn = False
        group_size = 16
    for i, filt in enumerate(filters):
        modules.append(nn.Conv2d(*filt))
        if (i != len(filters) - 1) or (not no_act_last_layer):
            if with_bn:
                modules.append(nn.BatchNorm2d(filt[1]))
            if with_gn:
                modules.append(nn.GroupNorm(math.ceil(filt[1]/float(group_size)), filt[1]))
            modules.append(activation)
            if dropout > 0:
                modules.append(nn.Dropout(p=dropout))
    modules = nn.Sequential(*modules)

    return modules


def build_mlp(filters, dropout=0., with_output_norm = False, with_bn=True,
              with_gn=False, no_act_last_layer=False, activation='relu'):
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'elu':
        activation = nn.ELU()
    else:
        raise NotImplementedError(f'Not supported activation function {activation}')
    if with_gn:  # mutually exclusive of gn and bn; bn default to True and will be overwritten by gn
        with_bn = False
        group_size = 16
    modules = nn.ModuleList()
    for i in range(len(filters)-1):
        modules.append(nn.Linear(filters[i], filters[i+1]))
        if not (no_act_last_layer and i == len(filters)-2):
            if with_bn:
                modules.append(nn.BatchNorm1d(filters[i+1]))
            if with_gn:
                modules.append(nn.GroupNorm(math.ceil(filters[i+1]/float(group_size)), filters[i+1]))
            modules.append(activation)
            if dropout > 0.:
                modules.append(nn.Dropout(p=dropout))
    if with_output_norm:
        modules.append(nn.Sigmoid())
    modules = nn.Sequential(*modules)

    return modules


def cvx_solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())

    solvers.options['show_progress'] = False

    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)

    return sol['x']
