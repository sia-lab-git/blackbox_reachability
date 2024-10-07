import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from utils import modules
from dynamics import dynamics
torch.cuda.empty_cache()
import argparse
torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

parser.add_argument('--USE_VALUE_NET', default=False, action='store_true',
                   help='use augmented dataset or not')

parser.add_argument('--value_net_folder_name', type=str, required=False, default="quadruped_pretrain", help='wandb project')

args = parser.parse_args()

USE_VALUE_NET=args.USE_VALUE_NET
value_net_folder_name=args.value_net_folder_name

dT=0.1

num_epochs = 15

class CustomDataset(Dataset):
    def __init__(self, quadruped_states):
        self.quadruped_states = quadruped_states

    def __len__(self):
        return self.quadruped_states.shape[0] # return a set of 

    def __getitem__(self, idx):
        return self.quadruped_states[idx, 0, :], self.quadruped_states[idx, 1:, :] 


cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

x_dim = 30
hidden_dim = 128
latent_dim = 2
ae_model = modules.AE(x_dim, hidden_dim, latent_dim, x_dim)
ae_model.to(DEVICE)
ae_model.load_state_dict(
        torch.load('hamiltonian_nn/quadruped/models/ae_model.pth', map_location='cpu'))
ae_model.eval()
for param in ae_model.parameters():
    param.requires_grad = False

if USE_VALUE_NET:
    quadruped_dynamics=dynamics.Quadruped(collisionR=0.5,set_mode='avoid',method='1AM3',ham_estimator_fname="ham_estimator_pretrained")
    value_net=modules.SingleBVPNet(in_features=10, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3,periodic_transform_fn=quadruped_dynamics.periodic_transform_fn)
    value_net.load_state_dict(torch.load(
        "./runs/%s/training/checkpoints/model_final.pth"%value_net_folder_name)["model"])
    for param in value_net.parameters(): 
            param.requires_grad = False
    value_net.cuda()
    value_net.eval()

def get_state_condensed(x):
    x_ex = x[...,:6].cuda()*1.0
    x_im =x[...,6:].cuda()*1.0
    x_im = torch.cat((x_im[...,29:33],x_im[...,-2:],x_im[...,:24]*0.1),dim=-1)
    z_im = ae_model.encoder(x_im)
    x_c=torch.cat((x_ex,z_im),dim=-1)
    return x_c


def collate_fn(batch_data):
    # compute x_c
    x=torch.cat([item[0][None, :] for item in batch_data], dim=0)
    x_c=get_state_condensed(x)
    # compute all next x_condensed
    next_x=torch.cat([item[1][None, :] for item in batch_data], dim=0)
    next_x_c=get_state_condensed(next_x)

    # sample dvdx
    dvds = torch.zeros(len(batch_data), 8).uniform_(-1, 1).cuda()
    
    if USE_VALUE_NET:
        # augment dataset
        rand_idx=torch.randperm(len(batch_data))
        idx = rand_idx[:int(len(batch_data)*0.99)]
        idx_c = rand_idx[int(len(batch_data)*0.99):]
        times= torch.zeros(idx.shape[0], 1).uniform_(0.0,0.6).cuda()
        x_c_shifted=x_c[idx,:]*1.0

        x_c_shifted[:,:2] = torch.zeros(idx.shape[0], 2).uniform_(-2,2).cuda()
        coords=torch.cat((times,x_c_shifted),dim=-1)
        # normalize to input
        inputs=quadruped_dynamics.coord_to_input(coords)
        # compute result
        model_results = value_net(
                        {'coords': inputs})

        dvds[idx,:] = quadruped_dynamics.io_to_dv(
                model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).detach().cuda()[...,1:]
        dvds[idx,:] *= (torch.randn_like(dvds[idx,:])*0.3+1.0) 
    
    dvds = torch.nn.functional.normalize(
                dvds, p=2, dim=-1).cuda()
    # compute best ham
    hams=[]
    valid_id=dvds[:,0]<10000 # all valid
    for i in range(6):
        f=(next_x_c[:,i,:]-x_c)/dT
        valid_id=torch.logical_and(torch.logical_and(valid_id,torch.norm(f[:,:2],dim=-1)<4.0),torch.abs(f[:,2])<3.0)  # robot sometimes being teleported! Filter these bad transitions out
        product=f*dvds
        product[...,6:]=torch.clip(product[...,6:],min=-0.1,max=0.1) # oppress less important states (x_low)
        if not USE_VALUE_NET: # for pretraining
            product[...,5]=torch.clip(product[...,5],min=-0.2,max=0.2)
        hams.append(torch.sum(product,dim=-1)[None,...])
    hams,_= torch.max(torch.cat(hams,dim=0),dim=0)
    valid_id=torch.logical_and(valid_id,torch.abs(hams)<7)  # robot sometimes being teleported! Filter these bad transitions out
    return torch.cat((x_c[valid_id,2:], dvds[valid_id,:]),dim=-1), hams[valid_id]


if __name__ == "__main__":
    quadruped_data = torch.from_numpy(
        np.load("data/data_collection_quadruped/quadruped_data.npy")).float()
    batch_size = 512
    data = CustomDataset(quadruped_states=quadruped_data)
    train_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print("loading pickle file completed, start training")

    model = modules.HamiltonianNetworkQuadruped(input_dim=14).cuda()
    # if USE_VALUE_NET:
    #     model.load_state_dict(
    #                 torch.load("./hamiltonian_nn/quadruped/models/ham_estimator_pretrained.pth", map_location='cpu'))
    model.cuda()
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    
    for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        model.train()
        train_loss = 0.0

        count=0
        pred_hams=[]
        gt_hams=[]
        for i, (input, labels) in tqdm(enumerate(train_loader), position=1, leave=False, colour='blue', ncols=80):

            optimizer.zero_grad()
            outputs = model(input)
            dvds=input[...,6:]
            error=outputs-labels
            if USE_VALUE_NET:
                error[error>0]*=2 # Mild modification: since the RL control is stochastic, learning error is inevitable. Here we shape ham pred slightly to the conservative end. This is not justified if the low-level policy is deterministic. Without this, Ham-NN still outperforms all baselines.
            loss = torch.mean(torch.abs(error))

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size

            count+=batch_size
            pred_hams.append(outputs)
            gt_hams.append(labels)
        gt_hams=torch.cat(gt_hams,dim=0).detach().cpu().numpy()
        pred_hams=torch.cat(pred_hams,dim=0).detach().cpu().numpy()
        
        scheduler.step()
        train_loss /= count
        print("loss: ",train_loss)
        if USE_VALUE_NET:
            torch.save(model.state_dict(), './hamiltonian_nn/quadruped/models/ham_estimator_quadruped.pth')
        else:
            torch.save(model.state_dict(), './hamiltonian_nn/quadruped/models/ham_estimator_pretrained.pth')

