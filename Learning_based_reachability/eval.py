import isaacgym

assert isaacgym
import torch
import numpy as np

from mini_gym.envs import *
from mini_gym.envs.base.legged_robot_config import Cfg
from mini_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
from mini_gym.envs.mini_cheetah.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
import random
import torch.nn as nn
from dynamics import dynamics
from utils import modules

def load_env(headless=False, num_envs=1):
    # prepare environment
    config_mini_cheetah(Cfg)

    from ml_logger import logger

    print(logger.glob("*"))
    print(logger.prefix)

    params = logger.load_pkl('parameters.pkl')

    if 'kwargs' in params[0]:
        deps = params[0]['kwargs']

        from mini_gym_learn.ppo.ppo import PPO_Args
        from mini_gym_learn.ppo.actor_critic import AC_Args
        from mini_gym_learn.ppo import RunnerArgs

        AC_Args._update(deps)
        PPO_Args._update(deps)
        RunnerArgs._update(deps)
        Cfg.terrain._update(deps)
        Cfg.commands._update(deps)
        Cfg.normalization._update(deps)
        Cfg.env._update(deps)
        Cfg.domain_rand._update(deps)
        Cfg.rewards._update(deps)
        Cfg.reward_scales._update(deps)
        Cfg.perception._update(deps)
        Cfg.domain_rand._update(deps)
        Cfg.control._update(deps)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = num_envs
    Cfg.terrain.num_rows = 2
    Cfg.terrain.num_cols = 2
    Cfg.terrain.border_size = 25

    from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=headless, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from mini_gym_learn.ppo.actor_critic import ActorCritic

    actor_critic = ActorCritic(
        num_obs=Cfg.env.num_observations,
        num_privileged_obs=Cfg.env.num_privileged_obs,
        num_obs_history=Cfg.env.num_observations * \
                        Cfg.env.num_observation_history,
        num_actions=Cfg.env.num_actions)
    # print(Cfg.env.num_observations ,Cfg.env.num_observation_history, "!!!!")
    print(logger.prefix)
    print(logger.glob("*"))
    weights = logger.load_torch("checkpoints/ac_weights_last.pt")
    actor_critic.load_state_dict(state_dict=weights)
    actor_critic.to(env.device)
    policy = actor_critic.act_inference

    return env, policy

def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

import time


class AE(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, latent_dim,output_dim):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
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
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded,encoded
    
class ControlNetwork(nn.Module):
    def __init__(self, input_dim=14,x_dim = 6,hidden_dim = 32,latent_dim = 2):
        super(ControlNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim+7, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(64, 64)
        self.dropout2 = nn.Dropout(0.1)
        # self.bn2 = nn.BatchNorm1d(32)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 6)
        
        self.ae_model_ = AE(x_dim, hidden_dim, latent_dim,x_dim)
        self.ae_model_.load_state_dict(
                torch.load('model.pth', map_location='cpu'))
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


def play_mc(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from mini_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    

    quadruped_dynamics=dynamics.Quadruped(collisionR=0.5,set_mode='avoid',method='NN')
    # load value net
    value_net=modules.SingleBVPNet(in_features=10, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3,periodic_transform_fn=quadruped_dynamics.periodic_transform_fn)
    value_net.load_state_dict(torch.load(
        "./runs/quadruped_exact/training/checkpoints/model_final.pth")["model"])
    for param in value_net.parameters():
            param.requires_grad = False
    value_net.cuda()

    # load autoencoder
    x_dim = 6
    hidden_dim = 32
    latent_dim = 2
    ae_model = AE(x_dim, hidden_dim, latent_dim,x_dim)
    ae_model.load_state_dict(
            torch.load('model.pth', map_location='cpu'))
    ae_model.cuda()
    ae_model.eval()
    for param in ae_model.parameters():
        param.requires_grad = False

    # load control estimator
    control_estimator = ControlNetwork(input_dim=14)
    control_estimator.load_state_dict(
            torch.load('control_estimator_quadruped_rl_policy.pth', map_location='cpu'))
    control_estimator.cuda()
    control_estimator.eval()
    for param in control_estimator.parameters():
        param.requires_grad = False

    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    print(recent_runs)

    logger.configure(Path(recent_runs[-1]).resolve())

    env, policy = load_env(headless=headless)
    print(env.dt)

    num_eval_steps = 500
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 3.0, 0.0, 0.0

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.zeros(num_eval_steps) 
    measured_y_vels = np.zeros(num_eval_steps)
    target_y_vels = np.zeros(num_eval_steps) 
    measured_yaw_vels = np.zeros(num_eval_steps)
    target_yaw_vels = np.zeros(num_eval_steps) 

    joint_positions = np.zeros((num_eval_steps, 12))

    

    tMax=0.5
    num_eval_steps = 100
    actions=torch.zeros(env.num_envs,12)

    while True:
        obs = env.reset()
        for i in tqdm(range(num_eval_steps)):

            # get coords
            base_pos_pre,base_quat_pre, base_lin_vel_pre,base_ang_vel_pre,net_contact_forces_pre,\
                    dof_pos_pre, dof_vel_pre, gravity_projected_pre = env.root_states[:,:2].clone(), \
                    env.base_quat.clone(), env.base_lin_vel[:,:2].clone(), env.base_ang_vel.clone(), \
                    env.net_contact_forces.clone(), env.dof_pos.clone(), env.dof_vel.clone(), env.projected_gravity.clone()
            
            contact_indicator=torch.abs(torch.sign(net_contact_forces_pre.reshape(env.num_envs,13,3)[:,3:13:3,-1]))

            _,_, yaw_pre= get_euler_xyz(base_quat_pre)
            yaw_pre[yaw_pre>torch.pi]-=torch.pi*2.0

            current_states=torch.cat([base_pos_pre.cpu(),yaw_pre[...,None].cpu(),base_lin_vel_pre.cpu(),base_ang_vel_pre[...,[2]].cpu(),
                                    dof_pos_pre.cpu(),dof_vel_pre.cpu(),gravity_projected_pre.cpu(),base_ang_vel_pre[...,:2].cpu(),
                                    contact_indicator.cpu(),actions.clone().cpu(),env.commands[:, [0,2]].clone().cpu()],dim=-1)
            x_im = torch.cat((current_states[...,35:39],current_states[...,-2:]),dim=-1).cuda()
            z_im = ae_model.encoder(x_im)
            states_condensed=torch.cat((current_states[...,:6].cuda(),z_im),dim=-1)
            states_condensed[...,:2]-=env.env_origins[...,:2] # shift the pos and clamp the input
            states_condensed=quadruped_dynamics.clip_state(states_condensed)
            times= torch.full((env.num_envs, 1), tMax).cuda()
            coords=torch.cat((times,states_condensed),dim=-1)
            
            # normalize to input
            inputs=quadruped_dynamics.coord_to_input(coords)
            # compute result
            model_results = value_net(
                            {'coords': inputs})

            values = quadruped_dynamics.io_to_value(
                model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))

            dvs = quadruped_dynamics.io_to_dv(
                model_results['model_in'], model_results['model_out'].squeeze(dim=-1))

            # compute opt control
            dvds = torch.nn.functional.normalize(
                    dvs[...,1:], p=2, dim=-1) 
            control_estimator_input=torch.cat((states_condensed[...,2:], dvds),dim=-1)
            outputs = control_estimator(control_estimator_input)
            _, predictions = torch.max(outputs, 1)

            # print(predictions)
            # change env command according to predictions
            env.commands[:, 0] = 3.0
            env.commands[predictions>=3, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = -2.0
            env.commands[predictions%3==1, 2] = 0.0
            env.commands[predictions%3==2, 2] = 2.0

            # update command
            obs['obs'][...,3:6]=env.commands[...,:3]* env.commands_scale
            with torch.no_grad():
                actions = policy(obs)

            obs, rew, done, info = env.step(actions)

            
            # contact_indicator=torch.abs(torch.sign(env.all_net_contact_forces.reshape(env.num_envs,14,3)[:,3:14:3,-1]))
            # # print(contact_indicator[1])
            print(env.root_states)

        # measured_x_vels[i] = env.base_lin_vel[0, 0]
        # target_x_vels[i] = x_vel_cmd
        # measured_y_vels[i] = env.base_lin_vel[0, 1]
        # target_y_vels[i] = y_vel_cmd
        # measured_yaw_vels[i] = env.base_ang_vel[0, 2]
        # target_yaw_vels[i] = yaw_vel_cmd

        # joint_positions[i] = env.dof_pos[0, :].cpu()



    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='blue', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='blue', linestyle="--", label="Desired")

    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='red', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_y_vels, color='red', linestyle="--", label="Desired")
    
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaw_vels, color='black', linestyle="-", label="Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_yaw_vels, color='black', linestyle="--", label="Desired")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_mc(headless=False)
