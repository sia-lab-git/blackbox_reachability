import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(sys.path[0], os.pardir), os.pardir)))
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
torch.cuda.empty_cache()
from utils import modules
from dynamics import dynamics
import argparse
torch.manual_seed(0)
np.random.seed(0)
parser = argparse.ArgumentParser()

parser.add_argument('--value_net_folder_name', type=str, required=False, default="quadruped_pretrain", help='wandb project')
args = parser.parse_args()
value_net_folder_name=args.value_net_folder_name
USE_VALUE_NET=True
class CustomDataset(Dataset):
    def __init__(self, quadruped_states):
        self.quadruped_states = quadruped_states.cuda()*1.0

    def __len__(self):
        return self.quadruped_states.shape[0] # return a set of 

    def __getitem__(self, idx):
        return self.quadruped_states[idx, 0, :], self.quadruped_states[idx, 1:, :] 


cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")
num_epochs = 10

x_dim = 30
hidden_dim = 128
latent_dim = 2
ae_model = modules.AE(x_dim, hidden_dim, latent_dim,x_dim)
ae_model.to(DEVICE)
ae_model.load_state_dict(
        torch.load('hamiltonian_nn/quadruped/models/ae_model.pth', map_location='cpu'))
ae_model.eval()
for param in ae_model.parameters():
    param.requires_grad = False

if USE_VALUE_NET:
    quadruped_dynamics=dynamics.Quadruped(collisionR=0.5,set_mode='avoid',method='NN',ham_estimator_fname="ham_estimator_quadruped")
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
    # compute all next x_c
    next_x=torch.cat([item[1][None, :] for item in batch_data], dim=0)
    next_x_c=get_state_condensed(next_x)
    # sample dvdx
    times= torch.zeros(x.shape[0], 1).uniform_(0.0,0.6).cuda()
    x_c_shifted=x_c*1.0

    x_c_shifted[:,:2] = torch.zeros(x.shape[0], 2).uniform_(-2,2).cuda()
    coords=torch.cat((times,x_c_shifted),dim=-1)
    # normalize to input
    inputs=quadruped_dynamics.coord_to_input(coords)
    # compute result
    model_results = value_net(
                    {'coords': inputs})

    dvds = quadruped_dynamics.io_to_dv(
            model_results['model_in'], model_results['model_out'].squeeze(dim=-1)).detach().cuda()[...,1:]
    dvds = torch.nn.functional.normalize(
                dvds, p=2, dim=-1).cuda()
    
    
    # generate control labels
    hams=[]
    valid_id=dvds[:,0]<10000 
    for i in range(6):
        f=(next_x_c[:,i,:]-x_c)/0.1
        valid_id=torch.logical_and(torch.logical_and(valid_id,torch.norm(f[:,:2],dim=-1)<4),torch.abs(f[:,2])<3.0)  # Robot sometimes being teleported! Filter these transitions out
        product=f*dvds
        # product[...,6:]=torch.clip(product[...,6:],min=-0.1,max=0.1) # oppress less important states (x_low)
        hams.append(torch.sum(product,dim=-1)[None,...])
    hams,control_labels= torch.max(torch.cat(hams,dim=0),dim=0)
    valid_id=torch.logical_and(valid_id,torch.abs(hams)<7)  # Robot sometimes being teleported! Filter these transitions out.
    return torch.cat((x_c[valid_id,2:], dvds[valid_id,:]),dim=-1),  control_labels[valid_id]






if __name__ == "__main__":

    quadruped_data = torch.from_numpy(
        np.load("data/data_collection_quadruped/quadruped_data.npy")).float()

    batch_size = 1024
    data = CustomDataset(quadruped_states=quadruped_data)
    train_loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    print("loading pickle file completed, start training")

    model = modules.ControlNetwork(input_dim=14)
    model.cuda()
    
    classes=['Accelerate left','Accelerate mid','Accelerate right','Decelerate left','Decelerate mid','Decelerate right']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.97)
    
    for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        model.train()
        train_loss = 0.0
        count=0
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for i, (input, labels) in tqdm(enumerate(train_loader), position=1, leave=False, colour='blue', ncols=80):

            optimizer.zero_grad()
            outputs = model(input)
            # print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
            count+=batch_size
            
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            # print(i,loss.item(),train_loss)
        print(' ')
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        
        scheduler.step()
        # print(outputs[:10],labels[:10])
        train_loss /= count
        print(train_loss)

        torch.save(model.state_dict(), 'hamiltonian_nn/quadruped/models/control_estimator_quadruped.pth') 
