from abc import ABC, abstractmethod
from utils import diff_operators
import math
import torch
import itertools

from utils.modules import HamiltonianNetworkQuadruped, AE, ControlNetwork, CAFCNet, HamiltonianNetworkSingleTrack6D, ControllerNetworkSingleTrack6D




class Dynamics(ABC):
    def __init__(self,
                 name: str, loss_type: str, set_mode: str,
                 state_dim: int, input_dim: int, nn_input_dim: int,
                 control_dim: int, disturbance_dim: int,
                 state_mean: list, state_var: list,
                 value_mean: float, value_var: float, value_normto: float,
                 deepReach_model: bool, method_: str, quaternion_start_dim: int):
        self.name = name
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.nn_input_dim = nn_input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model
        self.method_ = method_
        self.quaternion_start_dim = quaternion_start_dim

        assert self.loss_type in [
            'brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            'reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + \
                str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE
    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'reg':
            return (output * self.value_var / self.value_normto) + self.value_mean
        else:
            raise NotImplementedError

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'exact':
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepReach_model == 'reg':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        else:
            raise NotImplementedError
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            vis_term1 = (self.value_var / self.value_normto /
                         self.state_var.to(device=hes.device))**2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (self.value_var / self.value_normto /
                   self.state_var.to(device=hes.device))**2 * hes[..., 1:]

        return hes

    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def periodic_transform_fn(self, input):
        raise NotImplementedError

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def control_range(self, state):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds, dt):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

    def dsdt_(self, state, control, disturbance, ts):
        # freeze the dynamics if the next state exceed state boundary
        dsdt = self.dsdt(state, control, disturbance)
        time_up = (ts > 0)*1.0
        state_test_range_ = torch.tensor(self.state_test_range()).cuda()
        output1 = torch.any(state < state_test_range_[
                            :, 0]-0.01, -1, keepdim=False)
        output2 = torch.any(state > state_test_range_[
                            :, 1]+0.01, -1, keepdim=False)
        out_of_range_index = torch.logical_or(output1, output2)

        dsdt[out_of_range_index] = 0.0
        return dsdt*time_up

class BatchData:
    def __init__(self, x, dvdx, ham, opt_ctrl):
        self.x = x
        self.dvdx = dvdx
        self.ham = ham
        self.opt_ctrl = opt_ctrl

class SingleTrack6D(Dynamics):

    def __init__(self, set_mode: str, method: str, ham_estimator_fname: str):
        self.Izz = 2900
        self.m = 1964
        self.h = 0.47
        self.g = 9.81
        self.d_f = 1.4978
        self.d_r = 1.3722
        self.Cd0 = 241
        self.Cd1 = 25.1
        self.Cd2 = 0
        self.C_alpha_f = 60000/1.8
        self.C_alpha_r = 90000/1.8
        self.F_xf_acc = 0
        self.F_xr_acc = 1.0
        self.F_xf_brake = 0.6
        self.F_xr_brake = 0.4
        self.Fx_min = -18794
        self.Fx_max_cap = 5600
        self.L = self.d_f+self.d_r
        self.mu = 0.9
        self.mu_slide = 0.5
        self.delta_max = math.pi/10

       
        state_mean_=[0, 0, 0, 6, 0, 0]
        state_var_=[15, 15, math.pi, 6, 2, 2]
        if method == "NN":
             # load Hamiltonian approximation NN
            ham_nn_path = './hamiltonian_nn/slip_wheel_car/models/%s.pth'%ham_estimator_fname
            self.ham_net = HamiltonianNetworkSingleTrack6D(input_dim=10)
            self.ham_net.load_state_dict(
                torch.load(ham_nn_path, map_location='cpu'))
            self.ham_net.eval()
            self.ham_net.cuda()

            for param in self.ham_net.parameters():
                param.requires_grad = False
            try:
                self.control_estimator = ControllerNetworkSingleTrack6D(input_dim=10)
                self.control_estimator.load_state_dict(
                        torch.load('hamiltonian_nn/slip_wheel_car/models/opt_controller_singletrack6d.pth', map_location='cpu'))

                self.control_estimator.cuda()
                self.control_estimator.eval()
                for param in self.control_estimator.parameters():
                    param.requires_grad = False
            except Exception:
                pass
        elif method=="model_based":
            self.num_models=5
            label_std=torch.load("./dynamics_data/ensemble_models/label_std.pt")
            label_mean=torch.load("./dynamics_data/ensemble_models/label_mean.pt")
            self.dynamics_models=[CAFCNet(state_dim=6,control_dim=2,num_layers=3,num_neurons_per_layer=64,if_batch_norm=True,
                        inputs_mean=torch.as_tensor(state_mean_),inputs_std=torch.as_tensor(state_var_),
                        labels_mean=label_mean,labels_std=label_std,if_gpu=True) for i in range(self.num_models)]
            for i in range(self.num_models):
                self.dynamics_models[i].load_state_dict(
                    torch.load('./dynamics_data/ensemble_models/model%d.pth'%i, map_location='cpu'))
                self.dynamics_models[i].cuda()
                self.dynamics_models[i].training=False
                self.dynamics_models[i].eval()
                for param in self.dynamics_models[i].parameters():
                    param.requires_grad = False    


        super().__init__(
            name="SingleTrack6D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=6, input_dim=7, nn_input_dim=8, control_dim=2, disturbance_dim=0,
            state_mean=state_mean_,
            state_var=state_var_,
            value_mean=5,
            value_var=7,
            value_normto=0.02,
            deepReach_model='exact',
            method_=method,
            quaternion_start_dim=-1
        )

    def control_range(self, state):
        return [[-self.delta_max, self.delta_max], [self.Fx_min, self.Fx_max_cap]]

    def state_test_range(self):
        return [
            [-15, 15],
            [-15, 15],
            [-math.pi, math.pi],
            [0, 12],
            [-2, 2],
            [-2, 2]
        ]

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[2])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[2])
        transformed_input[..., 5:] = input[..., 4:]
        return transformed_input.cuda()


    def dsdt(self, state, control, disturbance):
        F_x = torch.minimum(control[..., 1], 75000/state[..., 3])
        F_x[torch.logical_and(F_x < 0, state[..., 3] <= 0)
            ] = 0  # can't brake if vx<0
        F_xf = F_x*self.F_xf_acc
        F_xf[F_x < 0] = F_x[F_x < 0]*self.F_xf_brake

        F_xr = F_x*self.F_xr_acc
        F_xr[F_x < 0] = F_x[F_x < 0]*self.F_xr_brake

        F_x_drag = -self.Cd0-self.Cd1 * \
            state[..., 3]-self.Cd2*torch.pow(state[..., 3], 2)

        # # really don't know how we compute F_yf before computing F_zf. So I remove the F_yf term
        # # F_x_tilde=F_xf*torch.cos(control[...,0])-F_yf*torch.sin(control[...,0])+F_xr
        F_x_tilde = F_xf*torch.cos(control[..., 0])+F_xr
        F_zf = (self.m*self.g*self.d_r-self.h*F_x_tilde)/self.L
        F_zr = (self.m*self.g*self.d_f+self.h*F_x_tilde)/self.L
        F_ymax_f = torch.sqrt(torch.clip(
            self.mu**2*torch.pow(F_zf, 2)-torch.pow(F_xf, 2), min=1e-5))
        F_ymax_r = torch.sqrt(torch.clip(
            self.mu**2*torch.pow(F_zr, 2)-torch.pow(F_xr, 2), min=1e-5))

        alpha_f = torch.arctan(
            (state[..., 4]+self.d_f*state[..., 5])/state[..., 3])-control[..., 0]
        alpha_r = torch.arctan(
            (state[..., 4]-self.d_r*state[..., 5])/state[..., 3])
        gamma_f = torch.abs(self.C_alpha_f*torch.tan(alpha_f)/3/F_ymax_f)
        gamma_r = torch.abs(self.C_alpha_r*torch.tan(alpha_r)/3/F_ymax_r)

        F_yf = -self.C_alpha_f*torch.tan(alpha_f)\
            * (1-gamma_f+1/3*gamma_f**2)

        F_yr = -self.C_alpha_r*torch.tan(alpha_r)\
            * (1-gamma_r+1/3*gamma_r**2)

        # if the rear tire starts to slide
        br = (torch.sqrt(F_xr**2+F_yr**2) > (F_zr*self.mu)).flatten()

        rear_tire_vx=state[..., br, 3]*1.0
        rear_tire_vy=state[..., br, 4]*1.0-state[..., br, 5]*self.d_r
        rear_tire_v=torch.sqrt(rear_tire_vx**2+rear_tire_vy**2)
        F_yr[..., br] = -F_zr[..., br]*self.mu_slide*rear_tire_vy / rear_tire_v
        F_xr[..., br] = -F_zr[..., br]*self.mu_slide*rear_tire_vx / rear_tire_v
            

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 3] * torch.cos(state[..., 2]) \
            - state[..., 4] * torch.sin(state[..., 2])
        dsdt[..., 1] = state[..., 3] * torch.sin(state[..., 2]) \
            + state[..., 4] * torch.cos(state[..., 2])
        dsdt[..., 2] = state[..., 5]
        dsdt[..., 3] = (F_xf*torch.cos(control[..., 0]) - F_yf*torch.sin(control[..., 0])
                        + F_xr+F_x_drag)/self.m + state[..., 5]*state[..., 4]
        dsdt[..., 4] = (F_yf*torch.cos(control[..., 0]) + F_yr
                        + F_xf*torch.sin(control[..., 0]))/self.m - state[..., 5]*state[..., 3]
        dsdt[..., 5] = (self.d_f*F_yf*torch.cos(control[..., 0]) +
                        self.d_f * F_xf*torch.sin(control[..., 0])-self.d_r*F_yr)/self.Izz

        # # sliding dynamics
        # # if the force exceed the limit, then the tire locked up


        bf = (torch.sqrt(F_xf**2+F_yf**2) > (F_zf*self.mu)).flatten()

        front_tire_vx=state[..., bf, 3]*1.0
        front_tire_vy=state[..., bf, 4]*1.0+state[..., bf, 5]*self.d_f
        front_tire_v=torch.sqrt(front_tire_vx**2+front_tire_vy**2)

        F_yf[..., bf] = -F_zf[..., bf]*self.mu_slide*front_tire_vy / front_tire_v
        F_xf[..., bf] = -F_zf[..., bf]*self.mu_slide*front_tire_vx / front_tire_v
        dsdt[..., bf, 3] = (F_xf[..., bf] + F_xr[..., bf]+F_x_drag[..., bf]) / \
            self.m + state[..., bf, 5]*state[..., bf, 4]
        dsdt[..., bf, 4] = (F_yf[..., bf] + F_yr[..., bf]) / \
            self.m - state[..., bf, 5]*state[..., bf, 3]
        dsdt[..., bf, 5] = (self.d_f*F_yf[..., bf] -
                           self.d_r*F_yr[..., bf])/self.Izz


        return dsdt
    
        # F_xr_world = F_xr*torch.cos(state[...,2])-F_yr*torch.sin(state[...,2])
        # F_yr_world = F_xr*torch.sin(state[...,2])+F_yr*torch.cos(state[...,2])

        # F_xf_world = F_xf*torch.cos(state[...,2]+control[...,0])-F_yf*torch.sin(state[...,2]+control[...,0])
        # F_yf_world = F_xf*torch.sin(state[...,2]+control[...,0])+F_yf*torch.cos(state[...,2]+control[...,0])
        # F_xf_world[..., bf] = F_xf[..., bf]*torch.cos(state[...,bf,2])-F_yf[..., bf]*torch.sin(state[...,bf,2])
        # F_yf_world[..., bf] = F_xf[..., bf]*torch.sin(state[...,bf,2])+F_yf[..., bf]*torch.cos(state[...,bf,2])
        # return dsdt, F_xr_world, F_yr_world, F_xf_world, F_yf_world, br, bf

    def boundary_fn(self, state):
        # Defining a Circular Target
        return torch.norm(state[..., :2], dim=-1) - 2.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds, dt):
        if self.method_ == "1AM3":
            possible_controls = list(itertools.product(
                [0, 1], repeat=self.control_dim))
            if self.set_mode == "avoid":
                ham = - torch.ones(
                    state.shape[0]).cuda()*torch.finfo().max
            else:
                ham = torch.ones(
                    state.shape[0]).cuda()*torch.finfo().max

            for possible_control in possible_controls:
                control_tensor = torch.ones(1,
                                            state.shape[1], self.control_dim)
                for j in range(self.control_dim):
                    control_tensor[...,
                                   j] *= self.control_range(state)[j][possible_control[j]]
                # clip F_x
                if possible_control[1] > 0:
                    control_tensor[..., 1] = torch.clip(
                        75000/state[..., 3], max=self.control_range(state)[1][1], min=0)

                next_state = state + dt*self.dsdt(
                    state, control_tensor.cuda(), torch.zeros_like(state).cuda())  # Rollout next state by assuming same "u" throughout

                f_est = (next_state-state)/dt
                ham_est = torch.sum(f_est.detach()*dvds, dim=-1)
                if self.set_mode == "avoid":
                    ham = torch.maximum(ham_est, ham)
                else:
                    ham = torch.minimum(ham_est, ham)
        elif self.method_ == "model_based":
            # for all vertexes, compute the ham, and pick the best one
            # print(state.shape)
            ham = - torch.ones(1,
                    state.shape[1]).cuda()*torch.finfo().max
            for i in range(2):
                for j in range(2):
                    f_preds=[]
                    u_samples = torch.zeros(state.shape[1],self.control_dim).cuda() # sample normalized control samples
                    u_samples[...,0]= 2.0*i-1
                    u_samples[...,1]= 2.0*j-1
                    input_=torch.cat((state[0],u_samples),dim=-1).cuda()
                    for model in self.dynamics_models:   
                        f_pred,f_1,f_2=model(input_) #B * STATEDIM
                        f_preds.append(f_pred[None,...])
                    f_preds=torch.cat(f_preds,dim=0)
                    d_bound,f_mean=torch.std_mean(f_preds,dim=0,keepdim=True)

                    ham_est = torch.sum(f_mean.detach()*dvds - torch.abs(d_bound.detach()*dvds), dim=-1)
                    ham = torch.maximum(ham_est, ham)


        elif self.method_ == "A3":
            # compute base hamiltonian
            dstb = 0
            base_control = torch.zeros(
                state.shape[1], self.control_dim).cuda()
            base_next_state = state + dt*self.dsdt(
                state, base_control, dstb)
            base_ham = torch.sum((base_next_state-state)/dt*dvds, dim=-1)
            # determine optimal control
            opt_control = torch.zeros(1,
                                      state.shape[1], self.control_dim).cuda()

            # determine opt delta
            control_tensor = torch.zeros(1,
                                         state.shape[1], self.control_dim).cuda()
            control_tensor[..., 0] = self.control_range(state)[0][1]

            next_state = state + dt*self.dsdt(
                state, control_tensor, dstb)
            f_est = (next_state-state)/dt
            ham_est = torch.sum(f_est.detach()*dvds, dim=-1)

            if self.set_mode == "avoid":
                opt_control[..., 0] = torch.sign(
                    ham_est-base_ham)*self.control_range(state)[0][1]
            else:
                opt_control[..., 0] = torch.sign(
                    ham_est-base_ham)*self.control_range(state)[0][0]

            # we need to take special care for determining opt Fx
            control_tensor_max = torch.zeros(1,
                                             state.shape[1], self.control_dim).cuda()
            control_tensor_max[..., 1] = torch.clip(
                75000/state[..., 3], max=self.control_range(state)[1][1], min=0)
            next_state_max = state + dt*self.dsdt(
                state, control_tensor_max, dstb)
            f_est_max = (next_state_max-state)/dt
            ham_est_max = torch.sum(f_est_max.detach()*dvds, dim=-1)

            control_tensor_min = torch.zeros(1,
                                             state.shape[1], self.control_dim).cuda()
            control_tensor_min[..., 1] = self.control_range(state)[1][0]
            next_state_min = state + dt*self.dsdt(
                state, control_tensor_min, dstb)
            f_est_min = (next_state_min-state)/dt
            ham_est_min = torch.sum(f_est_min.detach()*dvds, dim=-1)

            opt_control[..., 1] = self.control_range(state)[1][0]
            if self.set_mode == "avoid":
                opt_control[ham_est_max > ham_est_min, 1] = torch.clip(
                    75000/state[..., 3], max=self.control_range(state)[1][1], min=0)[ham_est_max > ham_est_min]
            else:
                opt_control[ham_est_max < ham_est_min, 1] = torch.clip(
                    75000/state[..., 3], max=self.control_range(state)[1][1], min=0)[ham_est_max < ham_est_min]

            # compute estimated ham
            opt_next_state = state + dt*self.dsdt(
                state, opt_control, dstb)
            ham = torch.sum((opt_next_state-state)/dt*dvds, dim=-1)
        elif self.method_ == "2M1":
            ham = torch.tensor([])
            n_control_samples = 100
            control_samples = torch.rand(
                n_control_samples, state.shape[1], 4).cuda()
            ctrl_range = self.control_range(state)
            for i in range(self.control_dim):
                control_samples[..., i] = control_samples[..., i] * \
                    (ctrl_range[i][1]-ctrl_range[i][0])+ctrl_range[i][0]

            for i in range(n_control_samples):
                # get next state here
                next_state = state + dt*self.dsdt(
                    state, control_samples[i, ...], None)

                f_est = ((next_state - state) / dt).detach()
                ham_sample = torch.sum(f_est.detach() * dvds, dim=-1)

                if ham.numel() == 0:
                    ham = ham_sample

                # try with max float
                if self.set_mode == "avoid":
                    ham = torch.maximum(ham_sample, ham)
                else:
                    ham = torch.minimum(ham_sample, ham)
        elif self.method_ == "NN":
            # assuming avoid problem
            dvds_mag = torch.norm(dvds, dim=-1)
            norm_dvds = dvds / dvds_mag.unsqueeze(-1)

            std_inputs = BatchData(state, norm_dvds, None, None)

            ham = self.ham_net(std_inputs)
            ham = ham * dvds_mag

        else:
            raise NotImplementedError
        return ham

    def optimal_control(self, state, dvds):
        dt = 0.001
        if self.method_ in (["1AM%d"%(i+1) for i in range(3)]+["model_based"]) :

            opt_control = torch.zeros(
                state.shape[0], self.control_dim).cuda()
            possible_controls = list(itertools.product(
                [0, 1], repeat=self.control_dim))
            if self.set_mode == "avoid":
                ham = - torch.ones(
                    state.shape[0]).cuda()*torch.finfo().max
            else:
                ham = torch.ones(
                    state.shape[0]).cuda()*torch.finfo().max

            for possible_control in possible_controls:
                control_tensor = torch.ones(
                    state.shape[0], self.control_dim).cuda()
                for j in range(self.control_dim):
                    control_tensor[...,
                                   j] *= self.control_range(state)[j][possible_control[j]]
                # clip F_x
                if possible_control[1] > 0:
                    control_tensor[..., 1] = torch.clip(
                        75000/state[..., 3], max=self.control_range(state)[1][1], min=0)

                next_state = state + dt*self.dsdt(
                    state, control_tensor.cuda(), torch.zeros_like(state).cuda())  # Rollout next state by assuming same "u" throughout
                f_est = (next_state-state)/dt
                ham_est = torch.sum(f_est*dvds, dim=-1)
                if self.set_mode == "avoid":
                    opt_control[ham_est > ham,
                                :] = control_tensor[ham_est > ham, :]
                    ham = torch.maximum(ham_est, ham)
                else:
                    opt_control[ham_est <= ham,
                                :] = control_tensor[ham_est <= ham, :]
                    ham = torch.minimum(ham_est, ham)

        elif self.method_ == "A3":
            # compute base hamiltonian
            dstb = 0
            base_control = torch.zeros(
                state.shape[0], self.control_dim).cuda()
            base_next_state = state + dt*self.dsdt(
                state, base_control, dstb)
            base_ham = torch.sum((base_next_state-state)/dt*dvds, dim=-1)
            # determine optimal control
            opt_control = torch.zeros(
                state.shape[0], self.control_dim).cuda()

            # determine opt delta
            control_tensor = torch.zeros(
                state.shape[0], self.control_dim).cuda()
            control_tensor[..., 0] = self.control_range(state)[0][1]

            next_state = state + dt*self.dsdt(
                state, control_tensor, dstb)
            f_est = (next_state-state)/dt
            ham_est = torch.sum(f_est*dvds, dim=-1)

            if self.set_mode == "avoid":
                opt_control[..., 0] = torch.sign(
                    ham_est-base_ham)*self.control_range(state)[0][1]
            else:
                opt_control[..., 0] = torch.sign(
                    ham_est-base_ham)*self.control_range(state)[0][0]

            # we need to take special care for determining opt Fx
            control_tensor_max = torch.zeros(
                state.shape[0], self.control_dim).cuda()
            control_tensor_max[..., 1] = torch.clip(
                75000/state[..., 3], max=self.control_range(state)[1][1], min=0)
            next_state_max = state + dt*self.dsdt(
                state, control_tensor_max, dstb)
            f_est_max = (next_state_max-state)/dt
            ham_est_max = torch.sum(f_est_max*dvds, dim=-1)

            control_tensor_min = torch.zeros(
                state.shape[0], self.control_dim).cuda()
            control_tensor_min[..., 1] = self.control_range(state)[1][0]
            next_state_min = state + dt*self.dsdt(
                state, control_tensor_min, dstb)
            f_est_min = (next_state_min-state)/dt
            ham_est_min = torch.sum(f_est_min*dvds, dim=-1)

            opt_control[..., 1] = self.control_range(state)[1][0]
            if self.set_mode == "avoid":
                opt_control[ham_est_max > ham_est_min, 1] = torch.clip(
                    75000/state[..., 3], max=self.control_range(state)[1][1], min=0)[ham_est_max > ham_est_min]
            else:
                opt_control[ham_est_max < ham_est_min, 1] = torch.clip(
                    75000/state[..., 3], max=self.control_range(state)[1][1], min=0)[ham_est_max < ham_est_min]
        elif self.method_=='NN':
            # norm_dvds = torch.nn.functional.normalize(
            #         dvds, p=2, dim=-1) 
            input_ = torch.cat((state[...,2:], dvds), dim=-1).unsqueeze(0)
            opt_control = self.control_estimator(input_).squeeze(0)

            # # unnormed control
            unnorm_control_ = opt_control*1.0
            control_range = [[-math.pi/10, math.pi/10], [-18794, 5600]]
            for i in range(2):
                unnorm_control_[..., i] = unnorm_control_[..., i] * \
                    (control_range[i][1]-control_range[i][0]) / \
                    20+(control_range[i][1]+control_range[i][0])/2
            return unnorm_control_

        return opt_control

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 12, 0, 0],
            'state_labels': ['px', 'py', r'$\psi$', 'vx', 'vy', r'$\delta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }
  

class Quadruped(Dynamics):
    def __init__(self, collisionR: float, set_mode: str, method: str, ham_estimator_fname: str):
        self.collisionR = collisionR

        self.dt=0.1
        
        # load the AE
        x_dim = 30
        hidden_dim = 128
        latent_dim = 2
        self.ae_model = AE(x_dim, hidden_dim, latent_dim, x_dim)
        self.ae_model.load_state_dict(
                torch.load('hamiltonian_nn/quadruped/models/ae_model.pth', map_location='cpu'))
        self.ae_model.cuda()
        self.ae_model.eval()
        for param in self.ae_model.parameters():
            param.requires_grad = False

        
        state_mean_=[0, 0, 0, 1.5, 0, 0, 0, 0]
        state_var_=[2.0, 2.0, math.pi, 1.5, 0.5, 1.5, 0.2, 0.2]
        if method =="NN":
            # load Hamiltonian approximation NN
            ham_nn_path = 'hamiltonian_nn/quadruped/models/%s.pth'%ham_estimator_fname
            self.ham_net = HamiltonianNetworkQuadruped()
            self.ham_net.load_state_dict(
                torch.load(ham_nn_path, map_location='cpu'))
            self.ham_net.eval()
            self.ham_net.cuda()
            for param in self.ham_net.parameters():
                param.requires_grad = False
            try:
                self.control_estimator = ControlNetwork(input_dim=14)
                self.control_estimator.load_state_dict(
                        torch.load('hamiltonian_nn/quadruped/models/control_estimator_quadruped.pth', map_location='cpu'))
                self.control_estimator.cuda()
                self.control_estimator.eval()
                for param in self.control_estimator.parameters():
                    param.requires_grad = False
            except Exception:
                pass

        elif method=="model_based":
            self.num_models=5
            label_std=torch.load("./data/ensemble_quadruped/label_std.pt")
            label_mean=torch.load("./data/ensemble_quadruped/label_mean.pt")
            self.dynamics_models=[CAFCNet(state_dim=8,control_dim=2,num_layers=3,num_neurons_per_layer=128,if_batch_norm=True,
                        inputs_mean=torch.as_tensor(state_mean_),inputs_std=torch.as_tensor(state_var_),
                        labels_mean=label_mean,labels_std=label_std,if_gpu=True) for i in range(self.num_models)]
            for i in range(self.num_models):
                self.dynamics_models[i].load_state_dict(
                    torch.load('./data/ensemble_quadruped/model%d.pth'%i, map_location='cpu'))
                self.dynamics_models[i].cuda()
                self.dynamics_models[i].training=False
                self.dynamics_models[i].eval()
                for param in self.dynamics_models[i].parameters():
                    param.requires_grad = False  

            

        super().__init__(
            name="Quadruped", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=8, input_dim=9, nn_input_dim=10, control_dim=2, disturbance_dim=0,
            state_mean=state_mean_,
            state_var=state_var_,
            value_mean=0.8,
            value_var=1.3,
            value_normto=0.02,
            deepReach_model='exact',
            method_=method,
            quaternion_start_dim=-1
        )

    def control_range(self, state):
        raise NotImplementedError

    def state_test_range(self):
        return [
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-math.pi, math.pi],
            [0, 3],
            [-0.5, -0.5],
            [-1.5, 1.5],
            [-0.2, 0.2],
            [-0.2, 0.2],
        ]

    def clip_state(self,state):
        return torch.clamp(state, torch.tensor(self.state_test_range(
            )).cuda()[..., 0], torch.tensor(self.state_test_range()).cuda()[..., 1]).detach()
    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 3] = (
            wrapped_state[..., 3] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]*1.0
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[2])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[2])
        transformed_input[..., 5:] = input[..., 4:]*1.0
        return transformed_input.cuda()

    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - 0.5
    
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds, dt, state_trainsitions=None):
        # assuming avoid problem
        if self.method_ =="NN":
            dvds_mag = torch.norm(dvds, dim=-1)
            norm_dvds = dvds / dvds_mag.unsqueeze(-1)
            std_inputs = torch.cat((state[...,2:], norm_dvds),dim=-1)
            ham = self.ham_net(std_inputs) 
            ham = ham * dvds_mag 

        elif self.method_ == "1AM3":
            current_state=state_trainsitions[...,0,:]
            ham = - torch.ones(1,
                    state.shape[1]).cuda()*torch.finfo().max
            for i in [1,3,4,6]:
                next_state=state_trainsitions[...,i,:]
                f_est=(next_state-current_state)/self.dt
                ham_est = torch.sum(f_est.detach()*dvds,dim=-1)
                ham = torch.maximum(ham_est, ham)

        elif self.method_ == "model_based":
            ham = - torch.ones(1,
                    state.shape[1]).cuda()*torch.finfo().max
            for i in range(2):
                for j in range(2):
                    f_preds=[]
                    u_samples = torch.zeros(state.shape[1],self.control_dim).cuda() # sample normalized control samples
                    u_samples[...,0]= 3.0*i # 0 or 3 
                    u_samples[...,1]= 4.0*j-2 # -2 or 2
                    input_=torch.cat((state[0],u_samples),dim=-1).cuda()
                    
                    for model in self.dynamics_models:   
                        model.training=False
                        f_pred,f_1,f_2=model(input_)
                        f_preds.append(f_pred[None,...])
                    
                    f_preds=torch.cat(f_preds,dim=0)
                    
                    f_std,f_mean=torch.std_mean(f_preds,dim=0,keepdim=True)
                    dbound=f_std*3
                    
                    ham_est = torch.sum(f_mean.detach()*dvds - torch.abs(dbound.detach()*dvds), dim=-1)
                    ham = torch.maximum(ham_est, ham)


        else:
            raise NotImplementedError
        return ham
    

    def get_condensed_state_input(self, state_full, env_org):
        x_im = torch.cat((state_full[...,35:39],state_full[...,-2:],state_full[...,6:30]*0.1),dim=-1).cuda()
        z_im = self.ae_model.encoder(x_im)
        states_condensed=torch.cat((state_full[...,:6].cuda(),z_im),dim=-1)

        states_condensed[...,:2]-=env_org[...,:2] # shift the pos and clamp the input
        return states_condensed



    def optimal_control(self, state, dvds):
        # dvds_mag = torch.norm(dvds, dim=-1)
        norm_dvds = torch.nn.functional.normalize(
                    dvds, p=2, dim=-1) 
        control_estimator_input=torch.cat((state[...,2:], norm_dvds),dim=-1)
        outputs = self.control_estimator(control_estimator_input)
        _, predictions = torch.max(outputs, 1)

        commands=torch.zeros(predictions.flatten().shape[0],2).cuda()
        commands[:, 0] = 3.0
        commands[predictions>=3, 0] = -0.

        commands[:, 1] = -2.0
        commands[predictions%3==1, 1] = 0.0
        commands[predictions%3==2, 1] = 2.0
        return commands


    def optimal_disturbance(self, state, dvds):
        return 0


    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 3, 0, 0, 0 ,0],
            'state_labels': ['px', 'py', r'$\psi$', 'vx', 'vy', r'$\delta$', 'z1', 'z2'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

    