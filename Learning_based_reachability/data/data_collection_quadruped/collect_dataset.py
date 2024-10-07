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
torch.manual_seed(0)
np.random.seed(0)
def load_env(headless=False):
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
    Cfg.env.num_envs = 1000
    Cfg.terrain.num_rows = 2
    Cfg.terrain.num_cols = 2
    Cfg.terrain.border_size = 25

    from mini_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=Cfg)
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
    print(Cfg.env.num_observations ,Cfg.env.num_observation_history, "!!!!")
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
def play_mc(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from mini_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    recent_runs = sorted(glob.glob(f"{MINI_GYM_ROOT_DIR}/runs/rapid-locomotion/*/*/*"), key=os.path.getmtime)
    print(recent_runs)

    logger.configure(Path(recent_runs[-1]).resolve())

    env, policy = load_env(headless=headless)
    print(env.dt)

    num_eval_steps = 1000
    x_vel_cmd, y_vel_cmd, yaw_vel_cmd = torch.zeros(env.num_envs), 0.0, torch.zeros(env.num_envs)

    measured_x_vels = np.zeros(num_eval_steps)
    target_x_vels = np.zeros(num_eval_steps) 
    measured_y_vels = np.zeros(num_eval_steps)
    target_y_vels = np.zeros(num_eval_steps) 
    measured_yaw_vels = np.zeros(num_eval_steps)
    target_yaw_vels = np.zeros(num_eval_steps) 


    measured_yaws = np.zeros(num_eval_steps)

    joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()

    state_transitions_dataset=[]
    for i in tqdm(range(num_eval_steps)):

        # print(env.base_lin_vel)
        if (i+1)%20==0:
            yaw_vel_cmd = torch.zeros(env.num_envs).uniform_(-2.0,2.0)
            x_vel_cmd = torch.zeros(env.num_envs).uniform_(0.0,3.3)
        
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd

        # update command
        # print(obs)
        # print(env.commands.shape,env.commands_scale.shape)
        # print(obs.shape,env.commands.shape,env.commands_scale.shape)
        obs['obs'][...,3:6]=env.commands[...,:3]* env.commands_scale

        with torch.no_grad():
            actions = policy(obs)
        # print(obs)
        obs, rew, done, info = env.step(actions)
        
        # # time.sleep(10)
        measured_x_vels[i] = env.base_lin_vel[0, 0]
        target_x_vels[i] = x_vel_cmd[0]
        measured_y_vels[i] = env.base_lin_vel[0, 1]
        target_y_vels[i] = y_vel_cmd
        measured_yaw_vels[i] = env.base_ang_vel[0, 2]
        target_yaw_vels[i] = yaw_vel_cmd[0]

        # get current state here
        base_pos_pre,base_quat_pre, base_lin_vel_pre,base_ang_vel_pre,net_contact_forces_pre,\
                dof_pos_pre, dof_vel_pre, gravity_projected_pre = env.root_states[:,:2].clone(), \
                env.base_quat.clone(), env.base_lin_vel[:,:2].clone(), env.base_ang_vel.clone(), \
                env.net_contact_forces.clone(), env.dof_pos.clone(), env.dof_vel.clone(), env.projected_gravity.clone()
        
        contact_indicator=torch.abs(torch.sign(net_contact_forces_pre.reshape(env.num_envs,13,3)[:,3:13:3,-1]))

        # compute yaw
        _,_, yaw_pre= get_euler_xyz(base_quat_pre)
        yaw_pre[yaw_pre>torch.pi]-=torch.pi*2.0
        current_state=torch.cat([base_pos_pre.cpu(),yaw_pre[...,None].cpu(),base_lin_vel_pre.cpu(),base_ang_vel_pre[...,[2]].cpu(),
                                 dof_pos_pre.cpu(),dof_vel_pre.cpu(),gravity_projected_pre.cpu(),base_ang_vel_pre[...,:2].cpu(),
                                 contact_indicator.cpu(),actions.clone().cpu(),env.commands[:, [0,2]].clone().cpu()],dim=-1)

        measured_yaws[i]=yaw_pre[0]
        # get all possible next states here
        all_next_states=[]
        
        for j in range(6):
            # update command here!
            if j<3:
                env.commands[:, 0] = 3.0
            else:
                env.commands[:, 0] = 0.0
            if j%3==0:
                env.commands[:, 2] = -2.0
            elif j%3==1:
                env.commands[:, 2] = 0.0
            else:
                env.commands[:, 2] = 2.0
            env.commands[:, 1] = 0
            new_obs=obs.copy()
            env.record_state()
            for k in range(5): # apply the action for 0.1 seconds
                new_obs['obs'][...,3:6]=env.commands[...,:3]* env.commands_scale
            
                # compute policy using the old observation + new command
                with torch.no_grad():
                    actions = policy(new_obs)
                new_obs,_,_,_=env.step(actions)
                # step the environment and undo
            base_pos,base_quat,base_lin_vel,base_ang_vel,contact_indicator,\
                dof_pos_post, dof_vel_post, gravity_projected=  env.undo() # default dt = 0.02s * 5

            # record the next state
            _,_, yaw_post= get_euler_xyz(base_quat)
            yaw_post[yaw_post>torch.pi]-=torch.pi*2.0
            next_state=torch.cat([base_pos[:,:2],yaw_post[...,None],base_lin_vel[:,:2],base_ang_vel[...,[2]],
                                    dof_pos_post,dof_vel_post,gravity_projected.cpu(),base_ang_vel[...,:2],
                                    contact_indicator,actions.clone().cpu(),env.commands[:, [0,2]].clone().cpu()],dim=-1)
            all_next_states.append(next_state[:,None,:])
            
            
            # reset obs in env
            env.obs_history=obs['obs_history'].clone()

        state_transitions=torch.cat([current_state[:,None,:]]+all_next_states,dim=1)
        
        state_transitions_dataset.append(state_transitions)
        
        joint_positions[i] = env.dof_pos[0, :].cpu()

    state_transitions_dataset=torch.cat(state_transitions_dataset,dim=0) # num_data * 7 * num_state_dim
    print(state_transitions_dataset.shape)
    np.save("data/data_collection_quadruped/quadruped_data.npy",state_transitions_dataset.detach().cpu().numpy())
    print("saved")
    # plot target and measured forward velocity
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='blue', linestyle="-", label="vx Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='blue', linestyle="--", label="vx Desired")

    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_y_vels, color='red', linestyle="-", label="vy Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_y_vels, color='red', linestyle="--", label="vy Desired")
    
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaw_vels, color='black', linestyle="-", label="omega Measured")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_yaw_vels, color='black', linestyle="--", label="omega Desired")
    axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_yaws, color='g', linestyle="--", label="yaw measured")
    axs[0].legend()
    axs[0].set_title("Forward Linear Velocity")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Velocity (m/s)")

    axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    axs[1].set_title("Joint Positions")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Joint Position (rad)")

    plt.tight_layout()
    plt.close(fig)
    # plt.figure(2)
    # plt.hist(measured_x_vels)
    # plt.figure(3)
    # plt.hist(measured_y_vels)
    # plt.figure(4)
    # plt.hist(measured_yaw_vels)
    plt.figure(2)
    plt.hist(state_transitions_dataset[:,0,2]) # yaw distribution
    plt.savefig("yaw_dist.png")
    plt.figure(3)
    plt.hist(state_transitions_dataset[:,0,3]) # vx distribution
    plt.savefig("vx_dist.png")
    plt.figure(4)
    plt.hist(state_transitions_dataset[:,0,4]) # vy distribution
    plt.savefig("vy_dist.png")
    plt.figure(5)
    plt.hist(state_transitions_dataset[:,0,5]) # omega distribution
    plt.savefig("vz_dist.png")


    

if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_mc(headless=False)
