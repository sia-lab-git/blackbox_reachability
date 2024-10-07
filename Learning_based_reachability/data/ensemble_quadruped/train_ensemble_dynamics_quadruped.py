import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
from utils import modules
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dynamics import dynamics
from random import randint
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

    valid_id=x_c[:,0]<10000 # all valid

    
    # random u
    i=randint(0,5)
    u=torch.zeros(x.shape[0],2).cuda()
    # random x,y in (-2,2)
    target_xy=torch.zeros_like(x_c[...,:2]).uniform_(-2,2)
    random_displacement=x_c[...,:2]-target_xy
    x_c[...,:2]-=random_displacement

    next_x_c[...,i,:2]-=random_displacement

    f=(next_x_c[:,i,:]-x_c)/0.1
    if i<3:
        u[:,0]=3.0
    
    if i%3==0:
        u[:,1]=-2.0
    elif i%3==2:
        u[:,1]=2.0
    valid_id=torch.logical_and(torch.logical_and(valid_id,torch.norm(f[:,:2],dim=-1)<4),torch.abs(f[:,2]<3))  # robot sometimes being teleported!

    return torch.cat((x_c[valid_id,:], u[valid_id,:]),dim=-1), f[valid_id,:].cuda()
    

if __name__ == "__main__":
    torch.set_printoptions(precision=2)
    quadruped_data = torch.from_numpy(
        np.load("data/data_collection_quadruped/quadruped_data.npy")).float()

    batch_size = 1000
    data = CustomDataset(quadruped_states=quadruped_data)
    train_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    fs=[]
    for i, (input, labels) in tqdm(enumerate(train_loader), position=1, leave=False, colour='blue', ncols=80):
        fs.append(labels)
    fs=torch.cat(fs,dim=0)
    label_std,label_mean=torch.std_mean(fs,dim=0,keepdim=True)
    torch.save(label_std,"./data/ensemble_quadruped/label_std.pt")
    torch.save(label_mean,"./data/ensemble_quadruped/label_mean.pt")

    # create seperate dataset
    dynamics_=dynamics.Quadruped(collisionR=0.5,set_mode="avoid", method="1AM2",ham_estimator_fname=None) # method doesn't matter since we only use dynamics_ to get state bound.
    input_mean,input_std=dynamics_.state_mean.clone(),dynamics_.state_var.clone()
    num_epochs = 1
    num_models = 5
    models=[modules.CAFCNet(state_dim=8,control_dim=2,num_layers=3,num_neurons_per_layer=128,if_batch_norm=True,
                inputs_mean=input_mean,inputs_std=input_std,labels_mean=label_mean,labels_std=label_std,if_gpu=True).cuda() for i in range(num_models)]
    optimizers = [optim.Adam(models[i].parameters(), lr=0.001) for i in range(num_models)]
    loss_fn = nn.MSELoss(reduction = 'sum')
    train_loaders=[]
    num_data=quadruped_data.shape[0]
    for i in range(num_models):
        models[i].train()
        idx_=torch.randperm(num_data)
        data = CustomDataset(quadruped_states=quadruped_data[idx_[:int(num_data*0.6)],...]) # every model use 60% of the dataset
        train_loaders.append(DataLoader(
            data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn))
        
    print("loading pickle file completed, start training")
    for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        train_losses = [0.0 for k in range(num_models)]
        for i in range(num_models):
            for j, (input_, labels) in tqdm(enumerate(train_loaders[i]), position=1, leave=False, colour='blue', ncols=80):
                models[i].training=True
                norm_labels=(labels - models[i].labels_mean) / models[i].labels_std 
                optimizers[i].zero_grad()
                f_pred =models[i](input_)
                loss = loss_fn(f_pred, norm_labels.cuda())

                # print(outputs.shape,labels.shape)
                loss.backward()
                optimizers[i].step()
                train_losses[i] += loss.item()

        
        for i in range(num_models):
            print(train_losses[i]/(j+1),end=' ')
        print(" ")

    
    # validate the output
    f_preds=[]
    f1_preds=[]
    f2_preds=[]
    num_val_labels=100000
    for i, (input_, labels) in tqdm(enumerate(train_loader), position=1, leave=False, colour='blue', ncols=80):
        if i>0:
            break
        f_labels=labels[100:110,...]
        input_samples=input_[100:110,...].cuda()
        u_samples=input_[100:110,-2:].cuda()
        for model in models:
            model.training=False
            model.eval()    
            f_pred,f_1,f_2=model(input_samples) #B * STATEDIM
            unnormed_f1=f_1*model.labels_std+model.labels_mean
            unnormed_f2=f_2*model.labels_std[...,None]

            f_from_affine=unnormed_f1+torch.einsum('bij,bjk->bik', unnormed_f2, u_samples[:, :, None].cuda()).squeeze(-1)
            assert (torch.abs(f_pred-f_from_affine)<1e-4).all() # check the two prediction are the same

            f_preds.append(f_pred[None,...])
            f1_preds.append(unnormed_f1[None,...])
            f2_preds.append(unnormed_f2[None,...])

        for i in range(num_models):
            torch.save(models[i].state_dict(), './data/ensemble_quadruped/model%d.pth'%i)
        f_preds=torch.cat(f_preds,dim=0)
        f1_preds=torch.mean(torch.cat(f1_preds,dim=0),dim=0)
        f2_preds=torch.mean(torch.cat(f2_preds,dim=0),dim=0)

        f_from_affine=unnormed_f1+torch.einsum('bij,bjk->bik', unnormed_f2, u_samples[:, :, None].cuda()).squeeze(-1)

        d_bound,f_mean=torch.std_mean(f_preds,dim=0,keepdim=True)
        # print("labels:",f_labels,"\n predictions:", f_pred,"\n predicion from affine dynamics:",f_from_affine,"\n3 sigma bound:",d_bound*3)

        print("labels:",f_labels,"\n predictions diff:", f_pred-f_labels,"\n3 sigma bound:",3*d_bound)
