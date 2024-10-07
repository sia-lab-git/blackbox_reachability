import torch
from torch.utils.data import Dataset
import numpy as np
import math


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.num_csl_samples = 0
        self.csl_dT = 0.0025
        self.num_vsl_samples = 0
        self.vsl_dT = 0.0025
        # self.num_discrete_samples = 10000  ##What is this for?
        self.num_discrete_samples = 10000  # What is this for?
        self.num_converged_samples = 0
        self.dT = 0.002

        if self.dynamics.name in ["Unitree_isaac","Unitree_full"]:
            print("Loading presaved dataset...")
            self.presaved_dataset = torch.load("dataset/unitree_stand")
            self.init_states=self.presaved_dataset[:,0,:]*1.0
            print("Loaded")
            self.p = self.numpoints/self.presaved_dataset.shape[0]
        elif self.dynamics.name == "Quadruped" and self.dynamics.method_!='NN':
            self.presaved_dataset = torch.from_numpy(
                np.load("data/data_collection_quadruped/quadruped_data.npy")).float()
            # filter out bad states
            valid_id=self.presaved_dataset[:,0,0]<10000 # all valid
            x=self.presaved_dataset[:,0,:]*1.0
            for i in range(6):
                next_x=self.presaved_dataset[:,i+1,:]*1.0
                f=(next_x-x)/0.1
                valid_id=torch.logical_and(torch.logical_and(valid_id,torch.norm(f[:,:2],dim=-1)<5.0),torch.abs(f[:,2])<3.0)  # robot sometimes being teleported!
            self.presaved_dataset=self.presaved_dataset[valid_id,...]

            self.init_states=self.presaved_dataset[:,0,:]*1.0
            

            self.p = self.numpoints/self.presaved_dataset.shape[0]
    def __len__(self):
        return 1

    def update_presaved_dataset(self,new_data):
        self.presaved_dataset = torch.cat([self.presaved_dataset,new_data],dim=0)
        self.p = self.numpoints/self.presaved_dataset.shape[0]

    def normalize_q(self, x):
        normalized_x = x*1.0
        q_tensor = x[..., self.dynamics.quaternion_start_dim:self.dynamics.quaternion_start_dim + 4]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2)  # normalize quaternion
        normalized_x[..., self.dynamics.quaternion_start_dim:
                     self.dynamics.quaternion_start_dim+4] = q_tensor
        return normalized_x
    
    def get_init_states(self):

        idx = torch.randperm(self.init_states.size(0))[:self.numpoints]
        model_states=torch.zeros(self.numpoints, 37)
        model_states=self.init_states[idx]
        return model_states

    def getitem_from_presaved_dataset(self):
        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), 0)
        else:
            if self.dynamics.name =="Unitree_isaac":
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, self.tMax)
            else:
                times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(
                    0, (self.tMax-self.tMin) * min((self.counter+1) / self.counter_end, 1.0))
                # times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, self.tMax)
            if self.dynamics.deepReach_model == 'reg':
                times[-self.num_src_samples:, 0] = 0.0

        
        idx = torch.randperm(self.presaved_dataset.size(0))[:int(self.presaved_dataset.size(0)*self.p)]
        if self.dynamics.name =="Unitree_isaac":
            model_states=torch.zeros(self.numpoints, 35)
            model_states[...,0]=self.presaved_dataset[idx][:,0,2]
            model_states[...,1:4]=self.presaved_dataset[idx][:,0,7:10]
            model_states[...,4:8]=self.presaved_dataset[idx][:,0,3:7]
            model_states[...,8:]=self.presaved_dataset[idx][:,0,10:]
        else:
            model_states=self.presaved_dataset[idx][:,0,:]
        states_transitions=self.presaved_dataset[idx]
        # if it is quadruped dynamics, we need to resample x,y position
        if self.dynamics.name == "Quadruped":
            target_xy=torch.zeros_like(model_states[...,:2]).uniform_(-1,1)
            # print(target_xy.shape)
            target_xy[...,0]*=self.dynamics.state_test_range()[0][1]
            target_xy[...,1]*=self.dynamics.state_test_range()[1][1]
            random_displacement=model_states[...,:2]-target_xy

            # get condensed state and randomly shift xy position
            model_states=self.dynamics.get_condensed_state_input(model_states,random_displacement.cuda()).cpu()

            states_transitions=self.dynamics.get_condensed_state_input(states_transitions,random_displacement.unsqueeze(1).repeat(1,7,1).cuda()).cpu()


        model_coords = torch.cat((times, model_states.float()), dim=1)
        boundary_values = self.dynamics.boundary_fn(model_coords[..., 1:])
        model_coords=self.dynamics.coord_to_input(model_coords)
        
        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around t=0
            dirichlet_masks = (model_coords[:, 0] == 0)

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords}, \
                {'states_transitions': states_transitions, 'boundary_values': boundary_values,'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.dynamics.name in ["Unitree_isaac", "Unitree_full"]:
            return self.getitem_from_presaved_dataset()
        elif self.dynamics.name == "Quadruped" and self.dynamics.method_!="NN":
            return self.getitem_from_presaved_dataset()
        
        # uniformly sample domain and include coordinates where source is non-zero
        model_states = torch.zeros(
            self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

        vsl_states = torch.zeros(
            self.num_vsl_samples, self.dynamics.state_dim).uniform_(-1, 1)
        vsl_times = self.vsl_dT + torch.zeros(self.num_vsl_samples, 1).uniform_(
            0, (self.tMax-self.vsl_dT) * min(self.counter / self.counter_end, 1.0))

        if self.dynamics.quaternion_start_dim >= 0:
            model_states = self.normalize_q(model_states)  # normalize q
            vsl_states = self.normalize_q(vsl_states)  # normalize q

        vsl_coords = torch.cat((vsl_times, vsl_states), dim=1)

        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(
                self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(
                self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), 0)
        else:
            # slowly grow time values from start time
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(
                0, (self.tMax-self.tMin) * min((self.counter+1) / self.counter_end, 1.0))
            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model == 'reg':
                times[-self.num_src_samples:, 0] = 0.0
        model_coords = torch.cat((times, model_states), dim=1)
        # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
        if self.dynamics.input_dim > self.dynamics.state_dim + 1:
            model_coords = torch.cat((model_coords, torch.zeros(
                self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        boundary_values = self.dynamics.boundary_fn(
            self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around t=0
            dirichlet_masks = (model_coords[:, 0] == 0)

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords, 'vsl_coords': vsl_coords}, \
                {'boundary_values': boundary_values,
                    'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords, 'vsl_coords': vsl_coords}, \
                {'boundary_values': boundary_values, 'reach_values': reach_values,
                    'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError


class CustomReachabilityDataset_Quadruped(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples
        self.num_csl_samples = 0
        self.csl_dT = 0.0025
        # self.num_discrete_samples = 10000  ##What is this for?
        self.num_discrete_samples = 10000  # What is this for?
        self.num_converged_samples = 0
        self.dT = 0.002
        # Path to the saved memory-mapped file
        load_path = 'datasets/deepreach_states_5e6_1.npy'
        # Load the memory-mapped array in read mode
        self.deepreach_states = np.lib.format.open_memmap(load_path, mode='r')

    def __len__(self):
        return 1

    def normalize_q(self, x):
        normalized_x = x*1.0
        q_tensor = x[..., self.dynamics.quaternion_start_dim:self.dynamics.quaternion_start_dim+4]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2)  # normalize quaternion
        normalized_x[..., self.dynamics.quaternion_start_dim:
                     self.dynamics.quaternion_start_dim+4] = q_tensor
        return normalized_x

    def sample_sin_cos(self, x):
        normalized_x = x*1.0
        theta1 = x[..., 2]*math.pi
        theta2 = x[..., 8]*math.pi

        normalized_x[..., 2] = torch.cos(theta1)
        normalized_x[..., 3] = torch.sin(theta1)

        normalized_x[..., 8] = torch.cos(theta2)
        normalized_x[..., 9] = torch.sin(theta2)
        return normalized_x

    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero
        # model_states = torch.zeros(
        #     self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)
        start_index = np.random.randint(
            0, self.deepreach_states.shape[0]-self.numpoints+1)
        model_states_total = torch.tensor(
            self.deepreach_states[start_index:start_index+self.numpoints, :], dtype=torch.float32)
        model_states = model_states_total[:, 0, :]
        self.dynamics.next_states = model_states_total[:, 1:, :]
        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(
                self.num_target_samples)
            model_states[-self.num_target_samples:] = self.dynamics.coord_to_input(torch.cat((torch.zeros(
                self.num_target_samples, 1), target_state_samples), dim=-1))[:, 1:self.dynamics.state_dim+1]

        if self.pretrain:
            # only sample in time around the initial condition
            times = torch.full((self.numpoints, 1), self.tMin)
        else:
            # slowly grow time values from start time
            times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(
                0, (self.tMax-self.tMin) * min((self.counter+1) / self.counter_end, 1.0))
            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model == 'reg':
                times[-self.num_src_samples:, 0] = 0.0
        model_coords = torch.cat((times, model_states), dim=1)
        # temporary workaround for having to deal with dynamics classes for parametrized models with extra inputs
        if self.dynamics.input_dim > self.dynamics.state_dim + 1:
            model_coords = torch.cat((model_coords, torch.zeros(
                self.numpoints, self.dynamics.input_dim - self.dynamics.state_dim - 1)), dim=1)

        boundary_values = self.dynamics.boundary_fn(
            self.dynamics.input_to_coord(model_coords)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(
                self.dynamics.input_to_coord(model_coords)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_coords.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_coords[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi':
            return {'model_coords': model_coords, 'start_index': start_index}, \
                {'boundary_values': boundary_values,
                    'dirichlet_masks': dirichlet_masks}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_coords': model_coords, 'start_index': start_index}, \
                {'boundary_values': boundary_values, 'reach_values': reach_values,
                    'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks}
        else:
            raise NotImplementedError
