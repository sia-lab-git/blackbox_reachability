import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import modules
from torch.optim.lr_scheduler import ExponentialLR
torch.cuda.empty_cache()
torch.manual_seed(0)
np.random.seed(0)

class BatchData:
    def __init__(self, x, dvdx, ham, opt_ctrl,dvds_mag=None,jacobian_Hu=None):
        self.x = x.cuda()
        self.dvdx = dvdx.cuda()
        self.ham = ham.cuda()
        self.opt_ctrl = opt_ctrl.cuda()
        self.dvds_mag=dvds_mag.cuda()
        self.jacobian_Hu=jacobian_Hu.cuda() # B * U


def collate_fn_(batch):

    x = torch.as_tensor(np.stack([item[0] for item in batch])).float()
    dvdx = torch.as_tensor(np.stack([item[1] for item in batch])).float()
    opt_ctrl = torch.as_tensor(
        np.stack([item[2] for item in batch])).float()
    ham = torch.as_tensor(np.stack([item[3] for item in batch])).float()

    dvds_mag = torch.as_tensor(np.stack([item[4] for item in batch]))
    jacobian_Hu = torch.as_tensor(np.stack([item[5] for item in batch]))
    return BatchData(x, dvdx, ham, opt_ctrl, dvds_mag, jacobian_Hu)




class CustomDataset(Dataset):
    def __init__(self, states, dvds, opt_ctrls, hams, dvds_mag=None, jacobian_H_u=None):
        self.states=states
        self.dvds=dvds
        self.opt_ctrls=opt_ctrls
        self.hams=hams
        self.dvds_mag=dvds_mag
        self.jacobian_H_u=jacobian_H_u

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        x = self.states[idx]
        dvds = self.dvds[idx]
        opt_control=self.opt_ctrls[idx]
        ham=self.hams[idx]
        dvds_mag=self.dvds_mag[idx]
        jacobian_H_u=self.jacobian_H_u[idx]
        return x, dvds, opt_control, ham, dvds_mag, jacobian_H_u



print("loading files")

state_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/state_data.npy")
dvds_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_data.npy")
opt_ctrl_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/opt_ctrl_data.npy")
ham_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/ham_data.npy")
dvds_mag_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_mag_data.npy")
jacobian_H_u_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/jacobian_H_u_data.npy")
    
data = CustomDataset(states=state_data, dvds=dvds_data, opt_ctrls=opt_ctrl_data, hams=ham_data, dvds_mag=dvds_mag_data, jacobian_H_u=jacobian_H_u_data)
    
batch_size = 1000
train_loader = DataLoader(data, batch_size=batch_size,
                          shuffle=True, collate_fn=collate_fn_)

print("loading files completed, start training")





def norm_control(control, control_range):
    norm_control_ = control*1.0
    for i in range(2):
        norm_control_[..., i] = (norm_control_[..., i]-(control_range[i][1] +
                                 control_range[i][0])/2)/(control_range[i][1]-control_range[i][0])*20
    return norm_control_


def unnorm_control(control, control_range):
    unnorm_control_ = control*1.0
    for i in range(2):
        unnorm_control_[..., i] = unnorm_control_[..., i] * \
            (control_range[i][1]-control_range[i][0]) / \
            20+(control_range[i][1]+control_range[i][0])/2
    return unnorm_control_


model = modules.ControllerNetworkSingleTrack6D(input_dim=10).cuda()
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = ExponentialLR(optimizer, gamma=0.9)
control_range = [[-math.pi/10, math.pi/10], [-18794, 5600]]
num_epochs = 30
for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
    model.train()
    train_loss = 0.0
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        coords = torch.cat((batch_data.x[...,2:], batch_data.dvdx*batch_data.dvds_mag[:, None]), dim=-1).unsqueeze(0)
        out=model(coords)
        outputs = unnorm_control(out*1.0,control_range)
        labels = batch_data.opt_ctrl
        # weighted_error=torch.sum(torch.abs((outputs.squeeze(0)-labels)* batch_data.jacobian_Hu),dim=-1)
        # weighted_error[batch_data.ham>0] = weighted_error[batch_data.ham>0]*0.1

        # normed_labels=norm_control(batch_data.opt_ctrl.cuda(),control_range)
        # weighted_error+=torch.sum(torch.abs(out.squeeze(0)-normed_labels),dim=-1)*0.001 # blend the normal loss
        # loss = torch.mean(weighted_error)
        normed_labels=norm_control(batch_data.opt_ctrl.cuda(),control_range)
        loss=torch.mean(torch.sum(torch.abs(out.squeeze(0)-normed_labels),dim=-1))


        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_size

    train_loss /= 4e5
    scheduler.step()


    print("loss: ", train_loss)
    torch.save(model.state_dict(), './hamiltonian_nn/slip_wheel_car/models/opt_controller_singletrack6d.pth')

