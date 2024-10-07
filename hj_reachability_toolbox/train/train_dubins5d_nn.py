import sys
sys.path.insert(0,'/home/vamsichilakamarri/siren-reach/')

from hamiltonian_nn.train_control_estimator_dubins5d_jax import ControllerNetwork, norm_control, unnorm_control
# Imports
import os
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import pickle
from flax import linen as nn
import time as tm

# Setup JAX
try:
    jax.config.update('jax_platform_name', 'cpu')
    jax.devices()
except:
    jax.config.update('jax_platform_name', 'cpu')
    jax.devices()

# Define the HamiltonianNetwork in JAX using Flax
class HamiltonianNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, dvdx):
        coords = jnp.concatenate((x, dvdx), axis=-1)

        x = jax.nn.relu(nn.Dense(64)(coords))
        x = jax.nn.relu(nn.Dense(64)(x))
        x = nn.Dense(1)(x)

        return jnp.squeeze(x, -1)

class Dubins5D_HAM(hj.Dynamics):

    def __init__(self,
                 aMax=2.,
                 wMax=2.,
                 L=1.,
                 max_position_disturbance=0,
                 control_mode="max",
                 disturbance_mode="min",
                 control_space=None,
                 disturbance_space=None):
        self.wMax = wMax
        self.aMax = aMax
        self.L = L
        self.dt = 0.1
        self.test_controls = jnp.array([[self.aMax,self.wMax], [-self.aMax,self.wMax], [self.aMax,-self.wMax], [-self.aMax,-self.wMax]])
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([- aMax, -wMax]),
                                        jnp.array([aMax, wMax]))
        self.control_range = jnp.array([[-aMax, aMax], [-wMax, wMax]])
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        
        self.ham_net = HamiltonianNetwork()
        self.ham_params = np.load('/home/vamsichilakamarri/siren-reach/hamiltonian_nn/ham_estimator_dubins5d_jax.npy', allow_pickle=True).item()
        self.control_net = ControllerNetwork()
        self.control_params = np.load('/home/vamsichilakamarri/siren-reach/hamiltonian_nn/control_estimator_dubins5d_jax.npy', allow_pickle=True).item()

        self.test_dvds = jnp.array([
            [1, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, -1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, -1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, -1],
        ])
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)


    def __call__(self, state, control, disturbance):
        """Implements the dynamics `dx_dt = f(x, u)"""
        _, _, v, psi, omega = state
        a, w = control
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), a, v/self.L*jnp.tan(omega), w])
    
    def next_state(self, state, control, disturbance, dt= None):
        if dt==None:
            dt = self.dt
        return (state + dt*self.__call__(state, control, disturbance))
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        dvds_mag = jnp.linalg.norm(grad_value, axis=-1)
        norm_dvds = grad_value / (dvds_mag+1e-5)

        # Prepare the inputs for the JAX Hamiltonian network
        std_inputs_x = jnp.asarray(state)
        std_inputs_dvdx = jnp.asarray(norm_dvds)

        # Compute the Hamiltonian using the JAX network
        control = unnorm_control(self.control_net.apply({'params': self.control_params}, std_inputs_x, std_inputs_dvdx),\
                                 self.control_range)
        return (jnp.clip(control,min=self.control_range[0],max=self.control_range[1]),0)
        

    def hamiltonian(self, state, time, value, grad_value):
        dvds_mag = jnp.linalg.norm(grad_value, axis=-1)
        norm_dvds = grad_value / dvds_mag

        # Prepare the inputs for the JAX Hamiltonian network
        std_inputs_x = jnp.asarray(state)
        std_inputs_dvdx = jnp.asarray(norm_dvds)

        # Compute the Hamiltonian using the JAX network
        ham = self.ham_net.apply({'params': self.ham_params}, std_inputs_x, std_inputs_dvdx)
        ham = ham * dvds_mag

        return ham
        # return jax.lax.fori_loop(0, 4, lambda i, ham: jnp.maximum(ham, grad_value @ (self.next_state(state, self.test_controls[i], 0) - state)/self.dt), -jnp.inf)
    
    # Solver prerequisites
    def open_loop_dynamics(self, state, time):
        _, _, v, psi, omega = state
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), 0., v/self.L*jnp.tan(omega), 0.])

    def control_jacobian(self, state, time):
        return jnp.array([
            [0., 0.],
            [0., 0.],
            [1., 0.],
            [0., 0.],
            [0., 1],
        ])

    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ])
    
    def updatefun(self, i, vals):
        state = vals[1]
        grad_value1 = self.test_dvds[2*i]
        grad_value2 = self.test_dvds[2*i + 1]

        dvds_mag = jnp.linalg.norm(grad_value1, axis=-1)
        norm_dvds = grad_value1 / dvds_mag

        # Prepare the inputs for the JAX Hamiltonian network
        std_inputs_x = jnp.asarray(state)
        std_inputs_dvdx = jnp.asarray(norm_dvds)

        # Compute the Hamiltonian using the JAX network
        ham1 = self.ham_net.apply({'params': self.ham_params}, std_inputs_x, std_inputs_dvdx)
        ham1 = ham1 * dvds_mag

        dvds_mag = jnp.linalg.norm(grad_value2, axis=-1)
        norm_dvds = grad_value2 / dvds_mag

        # Prepare the inputs for the JAX Hamiltonian network
        std_inputs_x = jnp.asarray(state)
        std_inputs_dvdx = jnp.asarray(norm_dvds)

        # Compute the Hamiltonian using the JAX network
        ham2 = self.ham_net.apply({'params': self.ham_params}, std_inputs_x, std_inputs_dvdx)
        ham2 = ham2 * dvds_mag

        # vals[0,i] = jnp.max(ham1,-ham2)
        max_val = jax.lax.max(jax.lax.abs(ham1), jax.lax.abs(ham2))
        vals = vals.at[0, i].set(max_val)
        return vals
    
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        vals = jax.lax.fori_loop(0,5, self.updatefun, jnp.vstack((jnp.zeros(5),state)))
        return vals[0]

if __name__ == "__main__":
    # Configure Solver
    dynamics = Dubins5D_HAM()
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=np.array([-5., -5., 0., -np.pi, -np.pi/6.]),
                                                                            hi=np.array([5., 5., 5., np.pi, np.pi/6.])),
                                                                   (31, 31, 21, 51, 11),
                                                                # (41, 41, 41, 51, 21),
                                                                periodic_dims=3)
    values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 2.5

    # Solver Set Up
    solver_settings = hj.SolverSettings.with_accuracy("high",hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)

    # Propagation time
    time = 0.
    target_time = -2.0 # Negative for BRT
    t_beg = tm.time()
    target_values = hj.step(solver_settings, dynamics, grid, time, values, target_time, False)
    t_end = tm.time()

    print(f"Execution Time(NN): {(t_end-t_beg)/60} mins")


    # Save files
    try:
        os.mkdir("dubins5d_nn")
    except:
        pass
    np.save("dubins5d_nn/target_values",target_values)
    with open("dubins5d_nn/grid.pkl", "wb") as f:
        pickle.dump(grid,f)