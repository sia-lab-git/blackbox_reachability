
import torch
import torch.nn as nn

import numpy as np
from torch.optim import Adam

from tqdm import tqdm
from torchvision.utils import save_image, make_grid
torch.manual_seed(0)
np.random.seed(0)
dataset_path = '~/datasets'

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")


batch_size = 10000

x_dim = 30
hidden_dim = 128
latent_dim = 2

lr = 1e-3

epochs = 30


from torch.utils.data import Dataset, DataLoader
import random

class CustomDataset(Dataset):
    def __init__(self, quadruped_states):
        # scale according to importance 0.1 for joint v and joint pos, 1.0 for g, foot contacts, and last command
        self.scale=torch.tensor([0.00001]*12+[0.00001]*12+[0.]*3+[0.]*2+[1.0]*4
                                +[0.]*12+[1]*2)
        self.quadruped_states = quadruped_states*self.scale
        
        
        
    def __len__(self):
        return self.quadruped_states.shape[0]

    def __getitem__(self, idx):
        return self.quadruped_states[idx, 0, :], self.quadruped_states[idx, random.randint(1, 6), :]


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
    

model = AE(x_dim, hidden_dim, latent_dim,x_dim)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(),
                             lr = lr,
                             weight_decay = 1e-8)

from torch.optim.lr_scheduler import ExponentialLR

b1 = 0.5
b2 = 0.999
scheduler = ExponentialLR(optimizer, gamma=0.9)
## Start training
iterations = 1


def collate_fn(batch_data):
    return torch.cat([item[0][None, :] for item in batch_data], dim=0), torch.cat([item[1][None, :] for item in batch_data], dim=0)


quadruped_data = torch.from_numpy(
    np.load("data/data_collection_quadruped/quadruped_data.npy")[..., 6:]).float()
data = CustomDataset(quadruped_states=quadruped_data)
train_loader = DataLoader(
    data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

BCE_loss = nn.MSELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(
        x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


print("Start training AE...")
model.train()
mse = nn.MSELoss()

from tqdm import tqdm
for epoch in range(epochs):
    overall_loss = 0

    for batch_idx, (x,x_next) in tqdm(enumerate(train_loader)):
        # print(x)
        x = x.view(batch_size, 47)
        x = x.to(DEVICE)
        x = torch.cat((x[...,29:33],x[...,-2:],x[...,:24]),dim=-1)
        

        x_next = x_next.view(batch_size, 47)
        x_next = x_next.to(DEVICE)
        x_next = torch.cat((x_next[...,29:33],x_next[...,-2:],x_next[...,:24]),dim=-1)
        if epoch>=2: #curriculum on q and q_dot
            if epoch==2:
                for g in optimizer.param_groups:
                    g['lr'] = 1e-4
            x[...,6:]*=10000
            x_next[...,6:]*=10000

        else:
            x[...,6:]*=0.0
            x_next[...,6:]*=0.0


        optimizer.zero_grad()



        x_hat, z=model(x)
        x_next_hat, z_next=model(x_next)
        # compute loss & backpropagation 
        # reconstruction_loss = mse(x, x_hat) + mse(x_next, x_next_hat) + mse(z,z_next)*0.1
        
        reconstruction_loss = mse(x, x_hat) + mse(x_next, x_next_hat) + mse(z,z_next)*0.3 \
              + mse(z,torch.zeros_like(z))*0.3 + mse(z_next,torch.zeros_like(z))*0.3
        # reconstruction_loss = mse(x, x_hat) + mse(x_next, x_next_hat) 
        overall_loss += reconstruction_loss.item()

        reconstruction_loss.backward()
        optimizer.step()
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ",
          overall_loss / (batch_idx*batch_size))
    scheduler.step()
    torch.save(model.state_dict(), 'hamiltonian_nn/quadruped/models/ae_model.pth')
print("Finish traning, start evaluation...")

model.eval()
data_test = CustomDataset(quadruped_states=quadruped_data[:5000, ...])
test_loader = DataLoader(
    data_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
with torch.no_grad():
    for batch_idx, (x,x_next) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, 47)
        x = x.to(DEVICE)
        x_next = x_next.view(batch_size, x_dim)
        x_next = x_next.to(DEVICE)
        
        x_hat, z=model(x)
        x_next_hat, z_next=model(x)
        print('x:', x[0, :])
        print('x_hat:', x_hat[0, :])
        break
    