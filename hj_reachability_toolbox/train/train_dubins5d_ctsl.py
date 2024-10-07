# Imports
import os
import jax
import jax.numpy as jnp
import numpy as np
import hj_reachability as hj
import pickle
import time as tm

# Setup JAX
try:
    jax.config.update('jax_platform_name', 'cpu')
    jax.devices()
except:
    jax.config.update('jax_platform_name', 'cpu')
    jax.devices()


# Configure Dubins5D System
class Dubins5D_CTSL(hj.Dynamics):

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
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(2), max_position_disturbance)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)


    def __call__(self, state, control, disturbance):
        """Implements the dynamics `dx_dt = f(x, u)"""
        _, _, v, psi, omega = state
        a, w = control
        return jnp.array([v * jnp.cos(psi), v * jnp.sin(psi), a, v/self.L*jnp.tan(omega), w])
    
    def next_state(self, state, control, disturbance, dt = None):
        if dt == None:
            dt = self.dt
        return (state + dt*self.__call__(state, control, disturbance))
    
    def optimal_control_and_disturbance(self, state, time, grad_value):
        raise NotImplementedError
        # Assume Dynamics are Affine

    def hamiltonian(self, state, time, value, grad_value):
        
        return jax.lax.fori_loop(0, 4, lambda i, ham: jnp.maximum(ham, grad_value @ (self.next_state(state, self.test_controls[i], 0) - state)/self.dt), -jnp.inf)
    
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
    
    def update_fun(self,i,vals):
        state = vals[1]
        f_est_mod = jax.lax.abs((self.next_state(state, self.test_controls[i], 0) - state)/self.dt)
        vals = vals.at[0].set(jax.lax.max(f_est_mod,vals[0]))
        return vals
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        vals =  jax.lax.fori_loop(0, 4, self.update_fun, jnp.vstack((jnp.zeros((5)),state)))
        return vals[0]    
    
# Configure Solver
dynamics = Dubins5D_CTSL()
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(hj.sets.Box(lo=np.array([-5., -5., 0., -np.pi, -np.pi/6.]),
                                                                           hi=np.array([5., 5., 5., np.pi, np.pi/6.])),
                                                               (31, 31, 21, 51, 11),
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
print(f"Execution Time(CTSL): {(t_end-t_beg)/60} mins")

# Save files
try:
    os.mkdir("dubins5d_ctsl")
except:
    pass
np.save("dubins5d_ctsl/target_values",target_values)
with open("dubins5d_ctsl/grid.pkl", "wb") as f:
    pickle.dump(grid,f)

