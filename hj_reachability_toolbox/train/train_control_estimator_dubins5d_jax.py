import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from jax import random, jit, grad
import pandas as pd
import numpy as np
from tqdm import tqdm


class CustomDataset:
    def __init__(self, dataframe):
        self.x = dataframe.loc[:, 0].to_numpy()
        self.y = dataframe.loc[:, 1].to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

def scene_collate_fn(batch):
    x = jnp.array([item[0]['state'] for item in batch])
    dvdx = jnp.array([item[0]['dvds'] for item in batch])
    opt_ctrl = jnp.array([item[1]['opt_ctrl'] for item in batch])
    ham = jnp.array([item[1]['ham'] for item in batch])
    return (x, dvdx, ham, opt_ctrl)

def create_batches(dataset, batch_size):
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch = [dataset[i] for i in batch_indices]
        yield scene_collate_fn(batch)

class ControllerNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, dvdx):
        coords = jnp.concatenate((x, dvdx), axis=-1)
        x = jax.nn.relu(nn.Dense(64)(coords))
        x = jax.nn.relu(nn.Dense(64)(x))
        x = nn.Dense(2)(x)
        return x

# Normalize control input
def norm_control(control, control_range):
    norm_control = control * 1.0
    for i in range(2):
        norm_control = norm_control.at[..., i].set(
            (control[..., i] - (control_range[i][1] + control_range[i][0]) / 2)
            / (control_range[i][1] - control_range[i][0]) * 20
        )
    return norm_control

# Un-normalize control input
def unnorm_control(norm_control, control_range):
    unnorm_control = norm_control * 1.0
    for i in range(2):
        unnorm_control = unnorm_control.at[..., i].set(
            norm_control[..., i] * (control_range[i][1] - control_range[i][0]) / 20
            + (control_range[i][1] + control_range[i][0]) / 2
        )
    return unnorm_control

def create_train_state(rng, learning_rate, input_dim):
    model = ControllerNetwork()
    dummy_x = jnp.ones((1, input_dim))
    dummy_dvdx = jnp.ones((1, input_dim))
    params = model.init(rng, dummy_x, dummy_dvdx)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jit
def train_step(state, batch, control_range):
    def loss_fn(params):
        x, dvdx, _, opt_ctrl = batch
        outputs = state.apply_fn({'params': params}, x, dvdx)
        labels = norm_control(opt_ctrl, control_range)
        loss = jnp.mean((outputs - labels) ** 2)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def eval_step(state, batch, control_range):
    x, dvdx, _, opt_ctrl = batch
    outputs = state.apply_fn({'params': state.params}, x, dvdx)
    labels = norm_control(opt_ctrl, control_range)
    loss = jnp.mean((outputs - labels) ** 2)
    return loss

if __name__ == "__main__":
    print("loading pickle file")
    unpickled_df = pd.read_pickle("./dubins5d_train.pkl")

    dataset = CustomDataset(dataframe=unpickled_df)
    batch_size = 100
    train_loader = create_batches(dataset, batch_size)

    unpickled_df = pd.read_pickle("./dubins5d_val.pkl")

    dataset_val = CustomDataset(dataframe=unpickled_df)
    val_loader = create_batches(dataset_val, batch_size)
    print("loading pickle file completed, start training")

    rng = random.PRNGKey(0)
    learning_rate = 0.0001
    input_dim = 5  
    control_range = [[-2, 2], [-2, 2]]

    state = create_train_state(rng, learning_rate, input_dim)

    num_epochs = 100
    for epoch in tqdm(range(num_epochs), position=0, desc="batch", leave=False, colour='green', ncols=80):
        train_loader = create_batches(dataset, batch_size)
        val_loader = create_batches(dataset_val, batch_size)
        
        # Training loop
        train_loss = 0.0
        for batch_data in train_loader:
            state, loss = train_step(state, batch_data, control_range)
            train_loss += loss

        train_loss /= len(dataset) // batch_size

        # Validation loop
        val_loss = 0.0
        for batch_data in val_loader:
            loss = eval_step(state, batch_data, control_range)
            val_loss += loss

        val_loss /= len(dataset_val) // batch_size
        print(f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Save the model
    np.save('control_estimator_dubins5d_jax.npy', state.params)
