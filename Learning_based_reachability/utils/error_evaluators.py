import torch
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.quaternion import get_euler_xyz

class Validator(ABC):
    @abstractmethod
    def validate(self, coords, values):
        raise NotImplementedError


class ValueThresholdValidator(Validator):
    def __init__(self, v_min, v_max):
        self.v_min = v_min
        self.v_max = v_max

    def validate(self, coords, values):
        return (values >= self.v_min)*(values <= self.v_max)


class MLPValidator(Validator):
    def __init__(self, mlp, o_min, o_max, model, dynamics):
        self.mlp = mlp
        self.o_min = o_min
        self.o_max = o_max
        self.model = model
        self.dynamics = dynamics

    def validate(self, coords, values):
        model_results = self.model(
            {'coords': self.dynamics.coord_to_input(coords.cuda())})
        inputs = torch.cat(
            (coords[..., 1:].cuda(), values[:, None].cuda()), dim=-1)
        outputs = torch.sigmoid(self.mlp(inputs).squeeze())
        return ((outputs >= self.o_min)*(outputs <= self.o_max)).to(device=values.device)


class MLPConditionedValidator(Validator):
    def __init__(self, mlp, o_levels, v_levels, model, dynamics):
        self.mlp = mlp
        self.o_levels = o_levels
        self.v_levels = v_levels
        self.model = model
        self.dynamics = dynamics
        assert len(self.o_levels) == len(self.v_levels) + 1

    def validate(self, coords, values):
        model_results = self.model(
            {'coords': self.dynamics.coord_to_input(coords.cuda())})
        inputs = torch.cat(
            (coords[..., 1:].cuda(), values[:, None].cuda()), dim=-1)
        outputs = torch.sigmoid(self.mlp(inputs).squeeze(
            dim=-1)).to(device=values.device)
        valids = torch.zeros_like(outputs)
        for i in range(len(self.o_levels) - 1):
            valids = torch.logical_or(
                valids,
                (outputs > self.o_levels[i])*(outputs <= self.o_levels[i+1]) *
                (values >= self.v_levels[i][0])*(values <= self.v_levels[i][1])
            )
        return valids


class MultiValidator(Validator):
    def __init__(self, validators):
        self.validators = validators

    def validate(self, coords, values):
        result = self.validators[0].validate(coords, values)
        for i in range(len(self.validators)-1):
            result = result * self.validators[i+1].validate(coords, values)
        return result


class SampleGenerator(ABC):
    @abstractmethod
    def sample(self, num_samples):
        raise NotImplementedError


class SliceSampleGenerator(SampleGenerator):
    def __init__(self, dynamics, slices):
        self.dynamics = dynamics
        self.slices = slices
        assert self.dynamics.state_dim == len(slices)

    def sample(self, num_samples):
        samples = torch.zeros(num_samples, self.dynamics.state_dim)
        for dim in range(self.dynamics.state_dim):
            if self.slices[dim] is None:
                samples[:, dim].uniform_(
                    *self.dynamics.state_test_range()[dim])
                samples[:, dim] = samples[:, dim]
            else:
                samples[:, dim] = self.slices[dim]
        return samples


# # get the tEarliest in [tMin:tMax:dt] at which the state is still valid
# def get_tEarliest(model, dynamics, state, tMin, tMax, dt, validator):
#     with torch.no_grad():
#         tEarliest = torch.full(state.shape[:-1], tMin - 1)
#         model_state = dynamics.normalize_state(state)

#         times_to_try = torch.arange(tMin, tMax + dt, dt)
#         for time_to_try in times_to_try:
#             blank_idx = (tEarliest < tMin)
#             time = torch.full((*state.shape[:-1], 1), time_to_try)
#             model_time = dynamics.normalize_time(time)
#             model_coord = torch.cat((model_time, model_state), dim=-1)[blank_idx]
#             model_result = model({'coords': model_coord.cuda()})
#             value = dynamics.output_to_value(output=model_result['model_out'][..., 0], state=state.cuda()).cpu()
#             valid_idx = validator.validate(torch.cat((time, state), dim=-1), value)
#             tMasked = tEarliest[blank_idx]
#             tMasked[valid_idx] = time_to_try
#             tEarliest[blank_idx] = tMasked
#             if torch.all(tEarliest >= tMin):
#                 break
#         blank_idx = (tEarliest < tMin)
#         if torch.any(blank_idx):
#             print(str(torch.sum(blank_idx)), 'invalid states')
#             tEarliest[blank_idx] = tMax
#         return tEarliest


def scenario_optimization(model,  dynamics, tMin, tMax, dt, set_type, control_type, scenario_batch_size, 
                          sample_batch_size, sample_generator, sample_validator, violation_validator, max_scenarios=None, 
                          max_samples=None, max_violations=None, tStart_generator=None, env=None, rl_policy=None):
    use_rollout_control=False
    if env is not None:
        sample_batch_size=env.num_envs
        scenario_batch_size=env.num_envs
        use_rollout_control=True
    if dynamics.method_=="NN" and dynamics.name=="Quadruped":
        use_rollout_control=False
    policy = model
    rem = ((tMax-tMin) / dt) % 1
    e_tol = 1e-12
    assert rem < e_tol or 1 - \
        rem < e_tol, f'{tMax-tMin} is not divisible by {dt}'
    assert tMax > tMin
    assert set_type in ['BRS', 'BRT']
    if set_type == 'BRS':
        print('confirm correct calculation of true values of trajectories (batch_scenario_costs)')
        raise NotImplementedError
    assert control_type in ['value', 'ttr', 'init_ttr']
    assert max_scenarios or max_samples or max_violations, 'one of the termination conditions must be used'
    if max_scenarios:
        assert (max_scenarios /
                scenario_batch_size) % 1 == 0, 'max_scenarios is not divisible by scenario_batch_size'
    if max_samples:
        assert (
            max_samples / sample_batch_size) % 1 == 0, 'max_samples is not divisible by sample_batch_size'

    # accumulate scenarios
    times = torch.zeros(0, )
    states = torch.zeros(0, dynamics.state_dim)
    values = torch.zeros(0, )
    costs = torch.zeros(0, )
    init_hams = torch.zeros(0, )
    mean_hams = torch.zeros(0, )
    mean_abs_hams = torch.zeros(0, )
    max_abs_hams = torch.zeros(0, )
    min_abs_hams = torch.zeros(0, )

    num_scenarios = 0
    num_samples = 0
    num_violations = 0

    pbar_pos = 0
    if max_scenarios:
        scenarios_pbar = tqdm(total=max_scenarios,
                              desc='Scenarios', position=pbar_pos)
        pbar_pos += 1
    if max_samples:
        samples_pbar = tqdm(total=max_samples,
                            desc='Samples', position=pbar_pos)
        pbar_pos += 1
    if max_violations:
        violations_pbar = tqdm(total=max_violations,
                               desc='Violations', position=pbar_pos)
        pbar_pos += 1

    nums_valid_samples = []
    while True:
        if (max_scenarios and (num_scenarios >= max_scenarios)) or (max_violations and (num_violations >= max_violations)):
            break

        batch_scenario_times = torch.zeros(scenario_batch_size, )
        batch_scenario_states = torch.zeros(
            scenario_batch_size, dynamics.state_dim)
        batch_scenario_values = torch.zeros(scenario_batch_size, )

        num_collected_scenarios = 0
        while num_collected_scenarios < scenario_batch_size:
            if max_samples and (num_samples >= max_samples):
                break
            # sample batch
            if tStart_generator is not None:
                batch_sample_times = tStart_generator(sample_batch_size)
                # need to round to nearest dt
                batch_sample_times = torch.round(batch_sample_times/dt)*dt
            else:
                batch_sample_times = torch.full((sample_batch_size, ), tMax)
            batch_sample_states = dynamics.equivalent_wrapped_state(
                sample_generator.sample(sample_batch_size))
            batch_sample_coords = torch.cat(
                (batch_sample_times.unsqueeze(-1), batch_sample_states), dim=-1)

            # validate batch
            with torch.no_grad():
                batch_sample_model_results = model(
                    {'coords': dynamics.coord_to_input(batch_sample_coords.cuda())})
                batch_sample_values = dynamics.io_to_value(batch_sample_model_results['model_in'].detach(
                ), batch_sample_model_results['model_out'].squeeze(dim=-1).detach())
            batch_valid_sample_idxs = torch.where(sample_validator.validate(
                batch_sample_coords, batch_sample_values))[0].detach().cpu()

            # store valid samples
            num_valid_samples = len(batch_valid_sample_idxs)
            start_idx = num_collected_scenarios
            end_idx = min(start_idx + num_valid_samples, scenario_batch_size)
            batch_scenario_times[start_idx:end_idx] = batch_sample_times[batch_valid_sample_idxs][:end_idx-start_idx]
            batch_scenario_states[start_idx:end_idx] = batch_sample_states[batch_valid_sample_idxs][:end_idx-start_idx]
            batch_scenario_values[start_idx:end_idx] = batch_sample_values[batch_valid_sample_idxs][:end_idx-start_idx]

            # update counters
            num_samples += sample_batch_size
            if max_samples:
                samples_pbar.update(sample_batch_size)
            num_collected_scenarios += end_idx - start_idx
            nums_valid_samples.append(num_valid_samples)
        if max_samples and (num_samples >= max_samples):
            break

        # propagate scenarios
        state_trajs = torch.zeros(scenario_batch_size, int(
            (tMax-tMin)/dt + 1), dynamics.state_dim)
        ctrl_trajs = torch.zeros(scenario_batch_size, int(
            (tMax-tMin)/dt), dynamics.control_dim)
        dstb_trajs = torch.zeros(scenario_batch_size, int(
            (tMax-tMin)/dt), dynamics.disturbance_dim)
        ham_trajs = torch.zeros(scenario_batch_size, int((tMax-tMin)/dt))

        

        if env is not None:
            obs=env.reset()
            actions=torch.zeros(env.num_envs,12).cuda()
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
            states_condensed=dynamics.get_condensed_state_input(current_states, env.env_origins)
            
            
            state_trajs[:, 0, :] = states_condensed
            # initialize everything correctly here!
            batch_scenario_states = states_condensed*1.0
            init_times = torch.full((scenario_batch_size, ), tMax).cuda()
            init_coords = torch.cat(
                (init_times.unsqueeze(-1), states_condensed), dim=-1)
            init_policy_results = policy(
                {'coords': dynamics.coord_to_input(init_coords)})
            batch_scenario_values=dynamics.io_to_value(init_policy_results['model_in'].detach(
                        ), init_policy_results['model_out'].squeeze(dim=-1).detach())

            
            
        else:
            state_trajs[:, 0, :] = batch_scenario_states

        for k in tqdm(range(int((tMax-tMin)/dt)), desc='Trajectory Propagation', position=pbar_pos, leave=False):
            if control_type == 'value':
                traj_time = tMax - k*dt
                traj_times = torch.full((scenario_batch_size, ), traj_time)
            # elif control_type == 'ttr':
            #     traj_times = get_tEarliest(model=model, dynamics=dynamics, state=state_trajs[:, k], tMin=tMin, tMax=traj_time, dt=dt, validator=sample_validator)
            # elif control_type == 'init_ttr':
            #     if k == 0:
            #         init_traj_times = get_tEarliest(model=model, dynamics=dynamics, state=state_trajs[:, k], tMin=tMin, tMax=traj_time, dt=dt, validator=sample_validator)
            #     traj_times = torch.maximum(init_traj_times - k*dt, torch.tensor(tMin)) # check whether this is the best thing to do for init_ttr

            traj_coords = torch.cat(
                (traj_times.unsqueeze(-1), state_trajs[:, k]), dim=-1)
            traj_policy_results = policy(
                {'coords': dynamics.coord_to_input(traj_coords.cuda())})
            traj_dvs = dynamics.io_to_dv(
                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()

            # TODO: I do not think there is actually any reason to store these trajs? Could save space by removing these.

            if use_rollout_control:
                # determine opt control by rolling out
                all_values=[]
                for j in range(6):
                    # update command
                    if j<3:
                        env.commands[:, 0] = 3.0
                    else:
                        env.commands[:, 0] = -0.
                    if j%3==0:
                        env.commands[:, 2] = -2.0
                    elif j%3==1:
                        env.commands[:, 2] = 0.0
                    else:
                        env.commands[:, 2] = 2.0
                    env.commands[:, 1] = 0
                    new_obs=obs.copy()
                    env.record_state()
                    for steps in range(1): # apply the action
                        new_obs['obs'][...,3:6]=env.commands[...,:3]* env.commands_scale
                        # compute policy using the old observation + new command
                        with torch.no_grad():
                            actions = rl_policy(new_obs)
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
                    possible_next_state_=dynamics.get_condensed_state_input(next_state, env.env_origins)
                    possible_next_state_ = torch.clamp(possible_next_state_, torch.tensor(dynamics.state_test_range(
                        )).cuda()[..., 0], torch.tensor(dynamics.state_test_range()).cuda()[..., 1])
                    possible_next_coords = torch.cat(
                        (traj_times.unsqueeze(-1).cuda(), possible_next_state_), dim=-1)
                    possible_next_results = policy(
                        {'coords': dynamics.coord_to_input(possible_next_coords)})
                    possible_next_values=dynamics.io_to_value(possible_next_results['model_in'].detach(
                        ), possible_next_results['model_out'].squeeze(dim=-1).detach())
                    all_values.append(possible_next_values[...,None])
                all_values=torch.cat(all_values,dim=-1)
                _,predictions=torch.max(all_values,dim=-1)
                commands=torch.zeros(predictions.flatten().shape[0],2).cuda()
                commands[:, 0] = 3.0
                commands[predictions>=3, 0] = 0.0
                commands[:, 1] = -2.0
                commands[predictions%3==1, 1] = 0.0
                commands[predictions%3==2, 1] = 2.0
                ctrl_trajs[:, k] =commands
                
            else:
                # determine opt control from the dynamics
                ctrl_trajs[:, k] = dynamics.optimal_control(
                    traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
            
            dstb_trajs[:, k] = dynamics.optimal_disturbance(
                traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda())
            try:
                ham_trajs[:, k] = dynamics.hamiltonian(
                    traj_coords[:, 1:].cuda(), traj_dvs[..., 1:].cuda(), dt) ##Added dt since required by HAM
            except:
                pass ## Added since HAM not required for discrete cases

            if tStart_generator is not None:  # freeze states whose start time has not been reached yet
                is_frozen = batch_scenario_times < traj_times
                is_unfrozen = torch.logical_not(is_frozen)
                state_trajs[is_frozen, k+1] = state_trajs[is_frozen, k]
                

                state_trajs[is_unfrozen, k+1] = dynamics.equivalent_wrapped_state(state_trajs[is_unfrozen, k].cuda() + dt*dynamics.dsdt(
                    state_trajs[is_unfrozen, k].cuda(), ctrl_trajs[is_unfrozen, k].cuda(), dstb_trajs[is_unfrozen, k].cuda()).cuda()).cpu()
                
            else:
                if env is not None:
                    # update command
                    env.commands[:,0]=ctrl_trajs[:, k,0]
                    env.commands[:,1]=0.0
                    env.commands[:,2]=ctrl_trajs[:, k,1]
                    obs['obs'][...,3:6]=env.commands[...,:3]* env.commands_scale
                    with torch.no_grad():
                        actions = rl_policy(obs)
                    obs, rew, done, info = env.step(actions)


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
                    next_state_=dynamics.get_condensed_state_input(current_states, env.env_origins)
                    next_state_ = torch.clamp(next_state_, torch.tensor(dynamics.state_test_range(
                        )).cuda()[..., 0], torch.tensor(dynamics.state_test_range()).cuda()[..., 1])
                else:
                    next_state_ = dynamics.equivalent_wrapped_state(state_trajs[:, k].cuda(
                        ) + dt*dynamics.dsdt(state_trajs[:, k].cuda(), ctrl_trajs[:, k].cuda(), dstb_trajs[:, k].cuda()))
                    next_state_ = torch.clamp(next_state_, torch.tensor(dynamics.state_test_range(
                        )).cuda()[..., 0], torch.tensor(dynamics.state_test_range()).cuda()[..., 1])
                state_trajs[:, k+1] = next_state_
        
        # compute batch_scenario_costs
        # TODO: need to handle the case of using tStart_generator when extending a trajectory by a frozen initial state will inadvertently affect cost computation (the min lx cost formulation is unaffected, but other cost formulations might care)
        if set_type == 'BRT':
            batch_scenario_costs = dynamics.cost_fn(state_trajs.cuda())
        elif set_type == 'BRS':
            if control_type == 'init_ttr':  # is this correct for init_ttr?
                batch_scenario_costs = dynamics.boundary_fn(
                    state_trajs.cuda())[:, (init_traj_times - tMin) / dt]
            elif control_type == 'value':
                batch_scenario_costs = dynamics.boundary_fn(
                    state_trajs.cuda())[:, -1]
            else:
                raise NotImplementedError  # what is the correct thing to do for ttr?

        # compute batch_scenario_init_hams, batch_scenario_mean_hams, batch_scenario_mean_abs_hams, batch_scenario_max_abs_hams, batch_scenario_min_abs_hams
        batch_scenario_init_hams = ham_trajs[:, 0]
        batch_scenario_mean_hams = torch.mean(ham_trajs, dim=-1)
        batch_scenario_mean_abs_hams = torch.mean(torch.abs(ham_trajs), dim=-1)
        batch_scenario_max_abs_hams = torch.max(
            torch.abs(ham_trajs), dim=-1).values
        batch_scenario_min_abs_hams = torch.min(
            torch.abs(ham_trajs), dim=-1).values

        # store scenarios
        times = torch.cat((times, batch_scenario_times.cpu()), dim=0)
        states = torch.cat((states, batch_scenario_states.cpu()), dim=0)
        values = torch.cat((values, batch_scenario_values.cpu()), dim=0)
        costs = torch.cat((costs, batch_scenario_costs.cpu()), dim=0)
        init_hams = torch.cat(
            (init_hams, batch_scenario_init_hams.cpu()), dim=0)
        mean_hams = torch.cat(
            (mean_hams, batch_scenario_mean_hams.cpu()), dim=0)
        mean_abs_hams = torch.cat(
            (mean_abs_hams, batch_scenario_mean_abs_hams.cpu()), dim=0)
        max_abs_hams = torch.cat(
            (max_abs_hams, batch_scenario_max_abs_hams.cpu()), dim=0)
        min_abs_hams = torch.cat(
            (min_abs_hams, batch_scenario_min_abs_hams.cpu()), dim=0)

        # update counters
        num_scenarios += scenario_batch_size
        if max_scenarios:
            scenarios_pbar.update(scenario_batch_size)
        num_new_violations = int(torch.sum(violation_validator.validate(
            batch_scenario_states, batch_scenario_costs)))
        num_violations += num_new_violations
        if max_violations:
            violations_pbar.update(num_new_violations)

    if max_scenarios:
        scenarios_pbar.close()
    if max_samples:
        samples_pbar.close()
    if max_violations:
        violations_pbar.close()

    violations = violation_validator.validate(states, costs)

    return {
        'times': times,
        'states': states,
        'values': values,
        'costs': costs,
        'init_hams': init_hams,
        'init_abs_hams': torch.abs(init_hams),
        'mean_hams': mean_hams,
        'mean_abs_hams': mean_abs_hams,
        'max_abs_hams': max_abs_hams,
        'min_abs_hams': min_abs_hams,
        'violations': violations,
        'valid_sample_fraction': torch.mean(torch.tensor(nums_valid_samples, dtype=float))/sample_batch_size,
        'violation_rate': 0 if not num_scenarios else num_violations / num_scenarios,
        'maxed_scenarios': (max_scenarios is not None) and num_scenarios >= max_scenarios,
        'maxed_samples': (max_samples is not None) and num_samples >= max_samples,
        'maxed_violations': (max_violations is not None) and num_violations >= max_violations,
        'batch_state_trajs': None if (max_samples and (num_samples >= max_samples)) else state_trajs,
        'batch_ctrl_trajs': None if (max_samples and (num_samples >= max_samples)) else ctrl_trajs,
    }


def target_fraction(model, dynamics, t, sample_validator, target_validator, num_samples, batch_size):
    with torch.no_grad():
        states = torch.zeros(0, dynamics.state_dim)
        values = torch.zeros(0, )

        while len(states) < num_samples:
            # sample batch
            batch_times = torch.full((batch_size, 1), t)
            batch_states = torch.zeros(batch_size, dynamics.state_dim)
            for dim in range(dynamics.state_dim):
                batch_states[:, dim].uniform_(
                    *dynamics.state_test_range()[dim])
                batch_states[:, dim] = batch_states[:, dim]
            batch_states = dynamics.equivalent_wrapped_state(batch_states)
            batch_coords = torch.cat((batch_times, batch_states), dim=-1)

            # validate batch
            batch_model_results = model(
                {'coords': dynamics.coord_to_input(batch_coords.cuda())})
            batch_values = dynamics.io_to_value(
                batch_model_results['model_in'], batch_model_results['model_out'].squeeze(dim=-1)).detach()
            batch_valids = sample_validator.validate(
                batch_coords, batch_values).detach().cpu()

            # store valid portion of batch
            states = torch.cat(
                (states, batch_states[batch_valids].cpu()), dim=0)
            values = torch.cat(
                (values, batch_values[batch_valids].cpu()), dim=0)

        states = states[:num_samples]
        values = values[:num_samples]
        coords = torch.cat((torch.full((num_samples, 1), t), states), dim=-1)
        valids = target_validator.validate(coords.cuda(), values.cuda())
    return torch.sum(valids) / num_samples


class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        s1 = int(2*input_size)
        s2 = int(input_size)
        s3 = int(input_size)
        self.l1 = torch.nn.Linear(input_size, s1)
        self.a1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(s1, s2)
        self.a2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(s2, s3)
        self.a3 = torch.nn.ReLU()
        self.l4 = torch.nn.Linear(s3, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        x = self.l4(x)
        return x
