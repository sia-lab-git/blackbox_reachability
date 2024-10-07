# while not reach num_samples:
#   randomly sample a set of x and dvds
#   normalize dvds
#   determine optimal control from a set of randomly sampled controls
#   add {(x,dvds),u*} to the dataset

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
import torch
from dynamics import dynamics
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
from utils import modules  
import argparse
torch.manual_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--USE_VALUE_NET', default=False, action='store_true',
                   help='use augmented dataset or not')

parser.add_argument('--value_net_folder_name', type=str, required=False, default="singleTrack6D_pretrained", help='wandb project')

args = parser.parse_args()

USE_VALUE_NET=args.USE_VALUE_NET
value_net_folder_name=args.value_net_folder_name

data_train = []
data_test = []
# num_points = 50000
num_points = 50000
n_control_samples = 10000
dt = 0.001
dynamics_ = dynamics.SingleTrack6D(set_mode="avoid", method="1AM3",ham_estimator_fname=None)
device = "cuda"


state_data=[]
dvds_data=[]
dvds_mag_data=[]
opt_ctrl_data=[]
ham_data=[]
jacobian_H_u_data=[]

if USE_VALUE_NET:
    value_net=modules.SingleBVPNet(in_features=8, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=256, num_hidden_layers=3,periodic_transform_fn=dynamics_.periodic_transform_fn)
    
    value_net.load_state_dict(torch.load(
        "./runs/%s/training/checkpoints/model_final.pth"%value_net_folder_name)["model"])
    value_net.cuda()
    value_net.eval()

for k in tqdm(range(20), position=0, desc="batch", leave=False, colour='green', ncols=80):
    # randomly sample a set of x and dvds
    states = dynamics_.input_to_coord(torch.zeros(1,
                                                  num_points, dynamics_.state_dim+1).uniform_(-1, 1))[..., 1:].to(device)
    if USE_VALUE_NET:
        times= torch.zeros(1,num_points, 1).uniform_(0.0,1.5).cuda()
        coords=torch.cat((times,states),dim=-1)
        # normalize to input
        inputs=dynamics_.coord_to_input(coords)
        # compute result
        model_results = value_net(
                        {'coords': inputs})

        dvds = dynamics_.io_to_dv(
                model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).detach().cuda()[...,1:]
        dvds *= (torch.randn_like(dvds)*0.1+1.0)  
        dvds_mag = torch.norm(dvds, dim=-1)

    else:
        dvds = torch.zeros(1,
                        num_points, dynamics_.state_dim).uniform_(-1, 1).to(device)
        
    # normalize dvds
    dvds = torch.nn.functional.normalize(
        dvds, p=2, dim=-1)  # normalize quaternion
    # determine optimal control from a set of randomly sampled controls
    ham = torch.tensor([])

    ctrl_range = dynamics_.control_range(states)

    opt_controls = torch.zeros(
        states.shape[1], dynamics_.control_dim).to(device)
    for i in tqdm(range(n_control_samples), position=1, desc="possibilities", leave=False, colour='blue', ncols=80):
        control_samples = torch.rand(
            states.shape[1], dynamics_.control_dim).to(device)
        for j in range(dynamics_.control_dim):
            control_samples[..., j] = control_samples[..., j] * \
                (ctrl_range[j][1]-ctrl_range[j][0])+ctrl_range[j][0]
        # get next states here
        next_state = states + dt*dynamics_.dsdt(
            states, control_samples.cuda(), None)

        f_est = ((next_state - states) / dt).detach()
        ham_sample = torch.sum(f_est * dvds, dim=-1)

        if ham.numel() == 0:
            ham = ham_sample
            opt_controls = control_samples

        else:
            opt_controls[(ham_sample > ham).squeeze(
                0)] = control_samples[(ham_sample > ham).squeeze(0)]

        ham = torch.maximum(ham_sample, ham)
    
    if USE_VALUE_NET:
        # numerically estimate Jacobian of ham on u
        jacobian_H_u=torch.zeros(1,num_points, dynamics_.control_dim).to(device)
        for i in range(2):
            control_p = opt_controls*1.0
            control_p[...,i]+=ctrl_range[i][1]/5.0
            state_p = states + dt*dynamics_.dsdt(
                states, control_p.cuda(), None)

            f_est = ((state_p - states) / dt).detach()
            ham_p = torch.sum(f_est * dvds, dim=-1)

            control_n = opt_controls*1.0
            control_n[...,i]-=ctrl_range[i][1]/5.0
            state_n = states + dt*dynamics_.dsdt(
                states, control_n.cuda(), None)

            f_est = ((state_n - states) / dt).detach()
            ham_n = torch.sum(f_est * dvds, dim=-1)

            jacobian_H_u[...,i]= (ham_p-ham_n)/ (ctrl_range[i][1]/2.5)
        for i in range(states.shape[1]):
                data_train.append([{"state": states[0, i, :].detach().cpu(), "dvds": dvds[0, i, :].detach().cpu(),"dvds_mag": dvds_mag[0,i].detach().cpu()}, {
                                "opt_ctrl": opt_controls[i, :].detach().cpu(), "ham": ham[0, i].detach().cpu(),"jacobian_Hu": jacobian_H_u[0, i, :].detach().cpu()}])
    
        jacobian_H_u_data.append(jacobian_H_u[0])
        dvds_mag_data.append(dvds_mag[0])
        
    state_data.append(states[0])
    dvds_data.append(dvds[0])
    opt_ctrl_data.append(opt_controls)
    ham_data.append(ham[0])
        
state_data=torch.cat(state_data,dim=0).detach().cpu().numpy()
dvds_data=torch.cat(dvds_data,dim=0).detach().cpu().numpy()
opt_ctrl_data=torch.cat(opt_ctrl_data,dim=0).detach().cpu().numpy()
ham_data=torch.cat(ham_data,dim=0).detach().cpu().numpy()
if USE_VALUE_NET:
    jacobian_H_u_data=torch.cat(jacobian_H_u_data,dim=0).detach().cpu().numpy()
    dvds_mag_data=torch.cat(dvds_mag_data,dim=0).detach().cpu().numpy()
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/state_data.npy",state_data)
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_data.npy",dvds_data)
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/opt_ctrl_data.npy",opt_ctrl_data)
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/ham_data.npy",ham_data)
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_mag_data.npy",dvds_mag_data)
    np.save("./hamiltonian_nn/slip_wheel_car/data/aug/jacobian_H_u_data.npy",jacobian_H_u_data)
    # df_train = pd.DataFrame(data_train)
    # df_train.to_pickle("./hamiltonian_nn/slip_wheel_car/data/singleTrack6d_train_aug.pkl")
else:
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/state_data.npy",state_data[:num_points*19,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/dvds_data.npy",dvds_data[:num_points*19,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/opt_ctrl_data.npy",opt_ctrl_data[:num_points*19,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/ham_data.npy",ham_data[:num_points*19,...])

    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/state_data_val.npy",state_data[num_points*19:,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/dvds_data_val.npy",dvds_data[num_points*19:,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/opt_ctrl_data_val.npy",opt_ctrl_data[num_points*19:,...])
    np.save("./hamiltonian_nn/slip_wheel_car/data/pretrain/ham_data_val.npy",ham_data[num_points*19:,...])
    # df_train = pd.DataFrame(data_train)
    # df_test = pd.DataFrame(data_test)
    # df_train.to_pickle("./hamiltonian_nn/slip_wheel_car/data/singleTrack6d_train.pkl")
    # df_test.to_pickle("./hamiltonian_nn/slip_wheel_car/data/singleTrack6d_val.pkl")
