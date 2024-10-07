import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import modules
torch.cuda.empty_cache()
import argparse
torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

parser.add_argument('--USE_VALUE_NET', default=False, action='store_true',
                   help='use augmented dataset or not')
args = parser.parse_args()
USE_VALUE_NET=args.USE_VALUE_NET

if USE_VALUE_NET:
    num_epochs = 30
else:
    num_epochs = 20

class BatchData:
    def __init__(self, x, dvdx, ham, opt_ctrl,dvds_mag=None,jacobian_Hu=None):
        self.x = x
        self.dvdx = dvdx
        self.ham = ham
        self.opt_ctrl = opt_ctrl
        if USE_VALUE_NET:
            self.dvds_mag=dvds_mag.cuda()
            self.jacobian_Hu=jacobian_Hu


def collate_fn_(batch):

    x = torch.as_tensor(np.stack([item[0] for item in batch])).float()
    dvdx = torch.as_tensor(np.stack([item[1] for item in batch])).float()
    opt_ctrl = torch.as_tensor(
        np.stack([item[2] for item in batch])).float()
    ham = torch.as_tensor(np.stack([item[3] for item in batch])).float()
    if USE_VALUE_NET:
        dvds_mag = torch.as_tensor(np.stack([item[4] for item in batch]))
        jacobian_Hu = torch.as_tensor(np.stack([item[5] for item in batch]))
        return BatchData(x, dvdx, ham, opt_ctrl, dvds_mag, jacobian_Hu)
    else:
        return BatchData(x, dvdx, ham, opt_ctrl)



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
        if USE_VALUE_NET:
            dvds_mag=self.dvds_mag[idx]
            jacobian_H_u=self.jacobian_H_u[idx]
            return x, dvds, opt_control, ham, dvds_mag, jacobian_H_u
        else:
            return x, dvds, opt_control, ham


if __name__ == "__main__":
    print("loading files")
    num_val_data=0
    if USE_VALUE_NET:
        state_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/state_data.npy")
        dvds_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_data.npy")
        opt_ctrl_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/opt_ctrl_data.npy")
        ham_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/ham_data.npy")
        dvds_mag_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/dvds_mag_data.npy")
        jacobian_H_u_data=np.load("./hamiltonian_nn/slip_wheel_car/data/aug/jacobian_H_u_data.npy")
    else:
        state_data=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/state_data.npy")
        dvds_data=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/dvds_data.npy")
        opt_ctrl_data=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/opt_ctrl_data.npy")
        ham_data=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/ham_data.npy")
        dvds_mag_data=None
        jacobian_H_u_data=None


        state_data_val=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/state_data_val.npy")
        dvds_data_val=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/dvds_data_val.npy")
        opt_ctrl_data_val=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/opt_ctrl_data_val.npy")
        ham_data_val=np.load("./hamiltonian_nn/slip_wheel_car/data/pretrain/ham_data_val.npy")
        num_val_data=state_data_val.shape[0]
    num_training_data=state_data.shape[0]
    data = CustomDataset(states=state_data, dvds=dvds_data, opt_ctrls=opt_ctrl_data, hams=ham_data, dvds_mag=dvds_mag_data, jacobian_H_u=jacobian_H_u_data)
    batch_size = 128
    train_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_)


    if not USE_VALUE_NET:
        data_val = CustomDataset(states=state_data_val, dvds=dvds_data_val, opt_ctrls=opt_ctrl_data_val, hams=ham_data_val)

        val_loader = DataLoader(data_val, batch_size=batch_size,
                                shuffle=True, collate_fn=collate_fn_)
    print("loading files completed, start training")
    model = modules.HamiltonianNetworkSingleTrack6D(input_dim=10).cuda()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        model.train()
        train_loss = 0.0
        for i, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch_data)
            labels = batch_data.ham.cuda()
            if USE_VALUE_NET:
                loss = criterion(outputs.flatten()*batch_data.dvds_mag, labels*batch_data.dvds_mag)
            else:
                loss = criterion(outputs.flatten(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size

        train_loss /= num_training_data

        model.eval()
        val_loss = 0.0
        print("training loss: ",train_loss)
        if not USE_VALUE_NET:
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    outputs = model(batch_data)
                    labels = batch_data.ham.cuda()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()*batch_size
                    if i == 0:
                        print("pred:", outputs[:2])
                        print("gt:", labels[:2])
            val_loss /= num_val_data
            print("validation loss: ", val_loss)

        if USE_VALUE_NET:
            torch.save(model.state_dict(), './hamiltonian_nn/slip_wheel_car/models/ham_estimator_singletrack6d.pth')
        else:
            torch.save(model.state_dict(), './hamiltonian_nn/slip_wheel_car/models/ham_estimator_singletrack6d_pretrain.pth')
