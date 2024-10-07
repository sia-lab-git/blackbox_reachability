import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

class HamiltonianNetworkSingleTrack6D(nn.Module):
    def __init__(self, input_dim=10):
        super(HamiltonianNetworkSingleTrack6D, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        # self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.2)
        # self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, batch):
        coords = torch.cat((batch.x[...,2:], batch.dvdx), dim=-1).cuda()
        x = torch.relu(self.fc1(coords))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x.squeeze(-1)
    
class ControllerNetworkSingleTrack6D(nn.Module):
    def __init__(self, input_dim=10):
        super(ControllerNetworkSingleTrack6D, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        # self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, coords):
        x = torch.relu(self.fc1(coords))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class HamiltonianNetworkQuadruped(nn.Module):
    def __init__(self, input_dim=14):
        super(HamiltonianNetworkQuadruped, self).__init__()
        self.fc1 = nn.Linear(input_dim+1, 256)
        self.bn2 = nn.BatchNorm1d(256)
        # self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(64, 64)
        # self.dropout2 = nn.Dropout(0.1)
        # self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        
 
    def forward(self, coords):
        cos=torch.cos(coords[...,0][...,None])
        sin=torch.sin(coords[...,0][...,None])
        coords=torch.cat((cos,sin,coords[...,1:]),dim=-1)
        x = torch.relu(self.fc1(coords))
        # x = self.dropout1(x)
        # x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = self.dropout2(x)
        x = self.fc4(x)

        return x.squeeze(-1)
    
class BatchLinear(nn.Linear):
    '''A linear layer'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(
            *[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class AE(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, latent_dim,output_dim):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, latent_dim)
        )
         
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
            # torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded,encoded
    
    
# # Define the neural network
# class ControlNetwork(nn.Module):
#     def __init__(self, input_dim=14):
#         super(ControlNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim+31, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(256, 256)
#         # self.fc3 = nn.Linear(64, 64)
#         self.dropout2 = nn.Dropout(0.1)
#         # self.bn2 = nn.BatchNorm1d(32)
        
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 6)


#     def forward(self, coords):
#         cos=torch.cos(coords[...,0][...,None])
#         sin=torch.sin(coords[...,0][...,None])
#         coords=torch.cat((cos,sin,coords[...,1:]),dim=-1)
#         x = torch.relu(self.fc1(coords))
#         # x = self.dropout1(x)
#         # x = torch.relu(self.bn2(self.fc2(x)))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         # x = self.dropout2(x)
#         x = self.fc4(x)

#         return x
class ControlNetwork(nn.Module):
    def __init__(self, input_dim=14):
        super(ControlNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim+31, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.1)
        # self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)
        
        self.ae_model_ = AE(30, 128, 2, 30)
        self.ae_model_.load_state_dict(
                torch.load('hamiltonian_nn/quadruped/models/ae_model.pth', map_location='cpu'))
        self.ae_model_.cuda()
        self.ae_model_.eval()

        for param in self.ae_model_.parameters():
            param.requires_grad = False

    def forward(self, coords):
        z_im=coords[...,4:6]*1.0
        x_im=self.ae_model_.decoder(z_im)
        cos=torch.cos(coords[...,0][...,None])
        sin=torch.sin(coords[...,0][...,None])
        coords=torch.cat((cos,sin,coords[...,1:],x_im),dim=-1)
        x = torch.relu(self.fc1(coords))
        # x = self.dropout1(x)
        # x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x = self.dropout2(x)
        x = self.fc4(x)

        return x

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        # freqs = torch.linspace(
        #     3, 300, steps=input.shape[-1])[None, None, :].to(input.device)
        # return torch.sin(freqs * input)
        return torch.sin(30 * input)


class Saturation(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.minimum(torch.maximum(input, torch.zeros_like(input)), torch.ones_like(input))


class FCBlock(nn.Module):
    '''A fully connected neural network.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        # Apply special initialization to first layer, if applicable.
        if first_layer_init is not None:
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords)
        return output
# class SingleBVPNet(nn.Module):
#     '''A canonical representation network for a BVP.'''

#     def __init__(self, out_features=1, type='sine', in_features=2,
#                  mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
#         super().__init__()
#         self.mode = mode
#         num_fourier_feature = int(hidden_features/2)
#         self.net = FCBlock(in_features=num_fourier_feature*2, out_features=out_features, num_hidden_layers=num_hidden_layers,
#                            hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
#         # self.fct = nn.Linear(1, 1)
#         self.B = torch.normal(mean=torch.zeros(
#             num_fourier_feature, in_features), std=1).cuda()
#         print(self)

#     def forward(self, model_input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         # Enables us to compute gradients w.r.t. coordinates
#         coords_org = model_input['coords'].clone(
#         ).detach().requires_grad_(True)

#         # obtain fourier embeddings
#         fourier_embeddinds = torch.cat((torch.sin(torch.matmul(self.B, coords_org.squeeze().T).T.unsqueeze(
#             0)), torch.cos(torch.matmul(self.B, coords_org.squeeze().T).T.unsqueeze(0))), dim=-1)

#         output = self.net(fourier_embeddinds)
#         return {'model_in': coords_org, 'model_out': output}


class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3,periodic_transform_fn=None, **kwargs):
        super().__init__()
        self.mode = mode
        self.periodic_transform_fn=periodic_transform_fn
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # self.fct = nn.Linear(1, 1)

        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone(
        ).detach().requires_grad_(True)
        coords_transformed=self.periodic_transform_fn(coords_org)
        output = self.net(coords_transformed)
        return {'model_in': coords_org, 'model_out': output}


class SingleBVPNetEval(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # self.fct = nn.Linear(1, 1)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(model_input['coords'])
        return {'model_in': model_input['coords'], 'model_out': output}


########################
# Initialization methods
def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
            # m.weight.uniform_(0.0, 0.0)


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(
                1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30,
                              np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            # m.weight.uniform_(0.0, 0.0)

class CAFCNet(torch.nn.Module):
    """
    Implements a fully connected control affine architecture f(x,u) = f1(x) + f2(x)u
    """
    def __init__(self, state_dim:int, control_dim:int, num_layers:int, num_neurons_per_layer:int, if_batch_norm:bool, inputs_mean, inputs_std, labels_mean, labels_std, if_gpu:bool):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.num_layers = num_layers # number of layers of f1(x) and f2(x)
        self.num_neurons_per_layer = num_neurons_per_layer
        
        assert type(if_batch_norm) is bool
        self.if_batch_norm = if_batch_norm
        
        assert type(if_gpu) is bool
        if if_gpu:
            if torch.cuda.is_available():
                self.gpu_device = 'cuda'
            else:
                self.gpu_device = torch.device("mps")
        else:
            self.gpu_device = 'cpu'

        self.inputs_mean = inputs_mean.to(self.gpu_device)
        self.inputs_std = inputs_std.to(self.gpu_device)
        self.labels_mean = labels_mean.to(self.gpu_device)
        self.labels_std = labels_std.to(self.gpu_device)

        self.nl = torch.nn.ReLU()
        self.f1_net_layers = []
        self.f2_net_layers = []
        for i in range(self.num_layers):
            if not i: # first layer
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.num_neurons_per_layer),
                    self.nl 
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.state_dim, self.num_neurons_per_layer),
                    self.nl 
                ))
                if self.if_batch_norm:
                    self.f1_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
                    self.f2_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
            elif i == self.num_layers - 1: # last layer
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.state_dim)
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.state_dim * self.control_dim)
                ))
            else:
                self.f1_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer),
                    self.nl
                ))
                self.f2_net_layers.append(torch.nn.Sequential(
                    torch.nn.Linear(self.num_neurons_per_layer, self.num_neurons_per_layer),
                    self.nl
                ))
                if self.if_batch_norm:
                    self.f1_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
                    self.f2_net_layers.append(torch.nn.Sequential(torch.nn.BatchNorm1d(num_features = self.num_neurons_per_layer)))
        self.f1_net = torch.nn.Sequential(*self.f1_net_layers)
        self.f2_net = torch.nn.Sequential(*self.f2_net_layers)

    def forward(self, inputs):
        state_inputs = inputs[:, :self.state_dim]
        state_inputs = (state_inputs - self.inputs_mean) / self.inputs_std # state input normalization
        if self.control_dim == 1:
            control_inputs = inputs[:, self.state_dim].unsqueeze(-1).unsqueeze(-1)
        else:
            control_inputs = inputs[:, self.state_dim:].unsqueeze(-1)
        
        f1_net_output = self.f1_net(state_inputs)
        f2_net_output = self.f2_net(state_inputs).view(-1, self.state_dim, self.control_dim)
        net_output = f1_net_output + torch.matmul(f2_net_output, control_inputs).squeeze(-1) # f(x, u) = f1(x) + f2(x)u

        if self.training: # train mode
            return net_output
        else:  # eval mode
            unnormalized_net_output = net_output * self.labels_std + self.labels_mean
            return unnormalized_net_output, f1_net_output, f2_net_output