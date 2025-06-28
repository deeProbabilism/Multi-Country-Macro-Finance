import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd

# Global configuration
tf.keras.backend.set_floatx('float64')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.makedirs('./ckpt', exist_ok=True)

# Global parameters
J = 5  # Number of countries
N_init = 100000  # Number of pretraining samples
N_interior = 8000 * J  # Number of interior samples
N_boundary_per_country = 2000  # Number of boundary samples per country
N_boundary = N_boundary_per_country * J  # Total number of boundary samples
N_total = N_interior + N_boundary  # Total number of samples
PRETRAIN_EPOCHS = [20, 20]  # List of epochs for pretraining phases
PRETRAIN_LEARNING_RATES = [1e-3, 1e-4]  # List of learning rates for pretraining phases
EPOCHS_MAIN = [100, 100, 100]  # List of epochs for main training phases
LEARNING_RATES_MAIN = [1e-4, 1e-5, 1e-6]  # List of learning rates for main training phases
rho = 0.03  # Time discount factor (ρ)
a = 0.1  # Productivity parameter
delta = 0.05  # Depreciation rate (δ)
sigma = 0.023  # Volatility parameter (σ)
psi = 5  # Investment adjustment parameter (ψ)
BATCH_SIZE = 512  # Batch size

# Boundary conditions
underline_omega = 0.2  # Lower bound for ω^i (expert wealth share)
underline_q = 1.29  # Value of q^i at the boundary when ω^i = underline_omega
bar_xi = tf.constant((a * psi + 1) / (psi * underline_q) - 1 / psi, dtype=tf.float64)

# PDE network definition
class PDE_Network(tf.keras.Model):
    def __init__(self):
        super(PDE_Network, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal()
        self.shared1 = tf.keras.layers.Dense(128, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64')
        self.shared2 = tf.keras.layers.Dense(128, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64')
        self.xi_tilde_layers = [tf.keras.layers.Dense(32, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64') for _ in range(J)]
        self.r_layer = tf.keras.layers.Dense(32, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64')
        self.xi_tilde_out = [tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64') for _ in range(J)]
        self.r_out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer, bias_initializer='zeros', dtype='float64')

    def call(self, x):
        x = self.shared1(x)
        x = self.shared2(x)
        xi_tilde = [self.xi_tilde_out[j](self.xi_tilde_layers[j](x)) for j in range(J)]
        xi_tilde = tf.concat(xi_tilde, axis=1)
        r = self.r_layer(x)
        r = self.r_out(r)
        return xi_tilde, r

# Pretraining network
class Initialize_PDE_Network(tf.keras.Model):
    def __init__(self):
        super(Initialize_PDE_Network, self).__init__()
        self.net = PDE_Network()

    def call(self, omega_init_batch, zeta_init_batch):
        inputs = tf.concat([omega_init_batch, zeta_init_batch], axis=1)
        xi_tilde, r = self.net(inputs)
        xi, q = compute_xi_q(omega_init_batch, zeta_init_batch, xi_tilde, rho, a, psi)
        sample_size = tf.cast(tf.shape(inputs)[0], tf.float64)
        loss = tf.reduce_sum((q - 1.3)**2) / sample_size + tf.reduce_sum((r - 0.03)**2) / sample_size
        boundary_loss = tf.constant(0.0, dtype=tf.float64)
        for j in range(J):
            mask = omega_init_batch[:, j] <= underline_omega
            boundary_loss += tf.reduce_sum(tf.where(mask, (xi_tilde[:, j] - bar_xi)**2, tf.zeros_like(xi_tilde[:, j])))
        boundary_loss /= sample_size
        loss += 1.0 * boundary_loss
        return loss

# Pretraining solver class
class Initialize_PDE_Solver(tf.keras.Model):
    def __init__(self, omega_init, zeta_init):
        super(Initialize_PDE_Solver, self).__init__()
        self.model = Initialize_PDE_Network()
        self.omega_init = omega_init
        self.zeta_init = zeta_init

    @tf.function
    def grad(self, omega_batch, zeta_batch):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(omega_batch, zeta_batch)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad, loss

    def train(self, epochs_list, learning_rate_list, batch_size):
        total_epochs = sum(epochs_list)
        current_epoch = 0
        dataset = tf.data.Dataset.from_tensor_slices((self.omega_init, self.zeta_init)).shuffle(1000).batch(batch_size, drop_remainder=True)
        
        with open('multi_c_pde.txt', 'a') as f:
            print('Pretraining history:', flush=True)
            f.write('Pretraining history:\n')
            for phase, (epochs, learning_rate) in enumerate(zip(epochs_list, learning_rate_list)):
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
                for epoch in range(epochs):
                    start_time = time.time()
                    train_loss = 0.0
                    num_batches = 0
                    for _, (omega_batch, zeta_batch) in enumerate(dataset):
                        grad, loss = self.grad(omega_batch, zeta_batch)
                        optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
                        train_loss += loss.numpy()
                        num_batches += 1
                    train_loss /= num_batches
                    elapsed = time.time() - start_time
                    current_epoch += 1
                    log_str = f'Phase {phase + 1}, Epoch {current_epoch}/{total_epochs}, Loss: {train_loss:.3e}, Time: {elapsed:.2f} seconds, Learning Rate: {learning_rate}'
                    print(log_str, flush=True)
                    f.write(log_str + '\n')
                    f.flush()
                    self.model.net.save_weights('./ckpt/pde_initialize_checkpoint')

# Dataset generator class
class DatasetGenerator:
    def __init__(self, N_interior, N_boundary_per_country, J, underline_omega):
        self.N_interior = N_interior
        self.N_boundary_per_country = N_boundary_per_country
        self.J = J
        self.underline_omega = underline_omega
        self.N_boundary = N_boundary_per_country * J
        self.N_total = N_interior + self.N_boundary

    def sample_interior_points(self):
        # Generate omega
        omega = np.random.uniform(self.underline_omega + 1e-6, 0.8, size=(self.N_interior, self.J)).astype(np.float64)  
        # Use Dirichlet distribution to generate zeta_full
        alpha = np.ones(self.J) * 50  # alpha parameters
        zeta_full = np.random.dirichlet(alpha, size=self.N_interior).astype(np.float64)
        zeta = zeta_full[:, :self.J-1]  # zeta is the first J-1 components of zeta_full
        boundary_country = np.full((self.N_interior, 1), -1, dtype=np.int32) # boundary_country is all -1, indicating interior points
        return np.concatenate([omega, zeta, boundary_country], axis=1)

    def generate_boundary_samples(self):
        boundary_samples = []
        for j in range(self.J):
            # Generate omega
            omega = np.random.uniform(self.underline_omega + 1e-6, 0.8, size=(self.N_boundary_per_country, self.J)).astype(np.float64)
            omega[:, j] = self.underline_omega  # Boundary condition  
            # Use Dirichlet distribution to generate zeta_full
            alpha = np.ones(self.J) * 50  # alpha parameters
            zeta_full = np.random.dirichlet(alpha, size=self.N_boundary_per_country).astype(np.float64)
            zeta = zeta_full[:, :self.J-1]  # zeta is the first J-1 components of zeta_full
            boundary_country = np.full((self.N_boundary_per_country, 1), j, dtype=np.int32) # boundary_country marks the boundary country
            samples = np.concatenate([omega, zeta, boundary_country], axis=1)
            boundary_samples.append(samples)
        Omega_boundary = np.concatenate(boundary_samples, axis=0)
        return Omega_boundary

    def create_dataset(self):
        Omega_interior = self.sample_interior_points()
        Omega_boundary = self.generate_boundary_samples()
        Omega_all = np.concatenate([Omega_interior, Omega_boundary], axis=0)
        is_interior = np.concatenate([np.ones(self.N_interior, dtype=bool), 
                                     np.zeros(self.N_boundary, dtype=bool)], axis=0)
        Omega_all = tf.convert_to_tensor(Omega_all, dtype=tf.float64)
        is_interior_all = tf.convert_to_tensor(is_interior, dtype=tf.bool)
        dataset = tf.data.Dataset.from_tensor_slices((Omega_all, is_interior_all))
        dataset = dataset.shuffle(buffer_size=self.N_total, reshuffle_each_iteration=True)
        return dataset

# Helper function: Compute ξ and q
def compute_xi_q(omega, zeta, xi_tilde, rho, a, psi):
    zeta_sum = tf.reduce_sum(zeta, axis=1, keepdims=True)
    zeta_J = 1.0 - zeta_sum
    zeta_full = tf.concat([zeta, zeta_J], axis=1)
    Xi = tf.reduce_sum(xi_tilde * zeta_full, axis=1, keepdims=True) + 1e-6
    xi = rho / Xi * xi_tilde
    q = (a * psi + 1) / (psi * xi + 1)
    return xi, q

def compute_q_and_sigma_q(Omega, net):
    Omega = tf.convert_to_tensor(Omega, dtype=tf.float64)
    N = tf.shape(Omega)[0]
    J = tf.shape(Omega)[1] // 2
    omega = Omega[:, :J]
    zeta = Omega[:, J:2*J-1]
    zeta_sum = tf.reduce_sum(zeta, axis=1, keepdims=True)
    zeta_J = 1.0 - zeta_sum
    zeta_full = tf.concat([zeta, zeta_J], axis=1)
    xi_tilde, _ = net(Omega[:, :-1])
    xi, q = compute_xi_q(omega, zeta, xi_tilde, rho, a, psi)

    def compute_grad(omega_sample):
        omega_sample = tf.reshape(omega_sample, [-1])
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(omega_sample)
            zeta_inner = omega_sample[J:2*J-1]
            zeta_sum_inner = tf.reduce_sum(zeta_inner, keepdims=True)
            zeta_J_inner = 1.0 - zeta_sum_inner
            zeta_full_inner = tf.concat([zeta_inner, zeta_J_inner], axis=0)
            xi_tilde_inner, _ = net(tf.expand_dims(omega_sample[:-1], axis=0))
            zeta_full_inner = tf.expand_dims(zeta_full_inner, axis=0)
            Xi_inner = tf.reduce_sum(xi_tilde_inner * zeta_full_inner, axis=1, keepdims=True) + 1e-6
            xi_inner = rho / Xi_inner * xi_tilde_inner
            q_inner = (a * psi + 1) / (psi * xi_inner + 1)
        q_grad_sample = tape.jacobian(q_inner, omega_sample)
        q_grad_sample = tf.squeeze(q_grad_sample, axis=0)
        del tape
        return q_grad_sample

    q_grad = tf.vectorized_map(compute_grad, Omega)
    sum_partial = tf.reduce_sum(q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :], axis=2)

    A_j_ta = tf.TensorArray(tf.float64, size=J, dynamic_size=False)
    b_j_ta = tf.TensorArray(tf.float64, size=J, dynamic_size=False)
    for j in range(J):
        A_j = q_grad[:, :, :J] * (1 - omega[:, tf.newaxis, :])
        zeta_deriv_part = q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :]
        zeta_deriv_part_padded = tf.pad(zeta_deriv_part, [[0,0], [0,0], [0,1]])
        A_j += zeta_deriv_part_padded
        A_j -= tf.einsum('bi,bj->bij', sum_partial, zeta_full)
        A_j -= tf.linalg.diag(q)
        zeta_j = zeta_full[:, j:j+1]
        omega_j = omega[:, j:j+1]
        b_j = sum_partial * zeta_j - q_grad[:, :, j] * (1 - omega_j)
        if j < J - 1:
            update_part = q_grad[:, :J-1, J + j] * zeta[:, j:j+1]
            b_j_updated = b_j[:, :J-1] - update_part
            b_j = tf.concat([b_j_updated, b_j[:, J-1:]], axis=1)
        A_j_ta = A_j_ta.write(j, A_j)
        b_j_ta = b_j_ta.write(j, b_j)

    A_j = tf.transpose(A_j_ta.stack(), [1, 0, 2, 3])
    b_j = tf.transpose(b_j_ta.stack(), [1, 0, 2])
    sigma_q = tf.linalg.solve(A_j, sigma * tf.expand_dims(b_j, axis=-1))[..., 0]
    return q, sigma_q

# Compute PDE loss with mask
def compute_pde_loss_with_mask(Omega, is_interior):
    Omega = tf.convert_to_tensor(Omega, dtype=tf.float64)
    N = tf.shape(Omega)[0]
    J = tf.shape(Omega)[1] // 2
    omega = Omega[:, :J]
    zeta = Omega[:, J:2*J-1]
    boundary_country = tf.cast(Omega[:, -1], tf.int32)
    zeta_sum = tf.reduce_sum(zeta, axis=1, keepdims=True)
    zeta_J = 1.0 - zeta_sum
    zeta_full = tf.concat([zeta, zeta_J], axis=1)
    xi_tilde, r = net(Omega[:, :-1])
    xi, q = compute_xi_q(omega, zeta, xi_tilde, rho, a, psi)

    def compute_grad_and_hess(omega_sample):
        omega_sample = tf.reshape(omega_sample, [-1])
        with tf.GradientTape(persistent=True) as tape_outer:
            tape_outer.watch(omega_sample)
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch(omega_sample)
                zeta_inner = omega_sample[J:2*J-1]
                zeta_sum_inner = tf.reduce_sum(zeta_inner, keepdims=True)
                zeta_J_inner = 1.0 - zeta_sum_inner 
                zeta_full_inner = tf.concat([zeta_inner, zeta_J_inner], axis=0) 
                xi_tilde_inner, r_inner = net(tf.expand_dims(omega_sample[:-1], axis=0))
                zeta_full_inner = tf.expand_dims(zeta_full_inner, axis=0) 
                Xi_inner = tf.reduce_sum(xi_tilde_inner * zeta_full_inner, axis=1, keepdims=True) + 1e-6
                xi_inner = rho / Xi_inner * xi_tilde_inner 
                q_inner = (a * psi + 1) / (psi * xi_inner + 1) 
            q_grad_sample = tape_inner.jacobian(q_inner, omega_sample)
        q_hess_sample = tape_outer.jacobian(q_grad_sample, omega_sample)
        q_grad_sample = tf.squeeze(q_grad_sample, axis=0)
        q_hess_sample = tf.squeeze(q_hess_sample, axis=0)
        return q_grad_sample, q_hess_sample

    q_grad, q_hess = tf.vectorized_map(compute_grad_and_hess, Omega)
    sum_partial = tf.reduce_sum(q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :], axis=2)
    A_j_ta = tf.TensorArray(tf.float64, size=J, dynamic_size=False)
    b_j_ta = tf.TensorArray(tf.float64, size=J, dynamic_size=False)
    for j in range(J):
        A_j = q_grad[:, :, :J] * (1 - omega[:, tf.newaxis, :])
        zeta_deriv_part = q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :]
        zeta_deriv_part_padded = tf.pad(zeta_deriv_part, [[0,0], [0,0], [0,1]])
        A_j += zeta_deriv_part_padded
        A_j -= tf.einsum('bi,bj->bij', sum_partial, zeta_full)
        A_j -= tf.linalg.diag(q)
        zeta_j = zeta_full[:, j:j+1]
        omega_j = omega[:, j:j+1]
        b_j = sum_partial * zeta_j - q_grad[:, :, j] * (1 - omega_j)
        if j < J - 1:
            update_part = q_grad[:, :J-1, J + j] * zeta[:, j:j+1]
            b_j_updated = b_j[:, :J-1] - update_part
            b_j = tf.concat([b_j_updated, b_j[:, J-1:]], axis=1)
        A_j_ta = A_j_ta.write(j, A_j)
        b_j_ta = b_j_ta.write(j, b_j)
    
    A_j = tf.transpose(A_j_ta.stack(), [1, 0, 2, 3])
    b_j = tf.transpose(b_j_ta.stack(), [1, 0, 2])
    sigma_q = tf.linalg.solve(A_j, sigma * tf.expand_dims(b_j, axis=-1))[..., 0]
    sigma_qK = sigma_q + tf.eye(J, batch_shape=[N], dtype=tf.float64) * sigma
    sigma_H = tf.reduce_sum(zeta_full[:, :, tf.newaxis] * sigma_qK, axis=1)
    mu_qK = -(a * psi + 1) / (psi * q) + 1.0 / psi + (1.0 / omega) * tf.reduce_sum(sigma_qK**2, axis=2) + r
    mu_H = tf.reduce_sum(zeta_full * mu_qK, axis=1, keepdims=True)
    mu_zeta = mu_qK[:, :J-1] - mu_H - tf.reduce_sum(sigma_H[:, tf.newaxis, :] * (sigma_qK[:, :J-1, :] - sigma_H[:, tf.newaxis, :]), axis=2)
    b = omega * ((a * psi + 1) / (psi * q) - 1.0 / psi - rho) + (1.0 / omega - 1)**2 * omega * tf.reduce_sum(sigma_qK**2, axis=2)
    
    mu_q_ta = tf.TensorArray(tf.float64, size=J, dynamic_size=False)
    for j in range(J):
        drift = tf.reduce_sum(q_grad[:, j, :J] * b, axis=1) + tf.reduce_sum(q_grad[:, j, J:2*J-1] * zeta * mu_zeta, axis=1)
        diffusion = 0.0
        omega_diag = tf.linalg.diag_part(q_hess[:, j, :J, :J])
        sigma_qK_sq_sum = tf.reduce_sum(sigma_qK**2, axis=2)
        diffusion += 0.5 * tf.reduce_sum(omega_diag * (1 - omega)**2 * sigma_qK_sq_sum, axis=1)
        sigma_omega_cross = tf.einsum('nhj,nlj->nhl', sigma_qK, sigma_qK)
        omega_factor = (1 - omega[:, :, tf.newaxis]) * (1 - omega[:, tf.newaxis, :])
        mask = 1.0 - tf.eye(J, batch_shape=[N], dtype=tf.float64)
        diffusion += tf.reduce_sum(q_hess[:, j, :J, :J] * omega_factor * sigma_omega_cross * mask, axis=[1, 2])
        zeta_diag = tf.linalg.diag_part(q_hess[:, j, J:2*J-1, J:2*J-1])
        sigma_zeta_diff = sigma_qK[:, :J-1, :] - sigma_H[:, tf.newaxis, :]
        sigma_zeta_diff_sq_sum = tf.reduce_sum(sigma_zeta_diff**2, axis=2)
        diffusion += 0.5 * tf.reduce_sum(zeta_diag * (zeta**2) * sigma_zeta_diff_sq_sum, axis=1)
        sigma_zeta_cross = tf.einsum('nhj,nlj->nhl', sigma_zeta_diff, sigma_zeta_diff)
        zeta_factor = zeta[:, :, tf.newaxis] * zeta[:, tf.newaxis, :]
        mask_zeta = 1.0 - tf.eye(J - 1, batch_shape=[N], dtype=tf.float64)
        diffusion += tf.reduce_sum(q_hess[:, j, J:2*J-1, J:2*J-1] * zeta_factor * sigma_zeta_cross * mask_zeta, axis=[1, 2])
        sigma_omega_zeta_cross = tf.einsum('nhj,nlj->nhl', sigma_qK[:, :, :], sigma_zeta_diff)
        omega_zeta_factor = (1 - omega[:, :, tf.newaxis]) * zeta[:, tf.newaxis, :]
        diffusion += tf.reduce_sum(q_hess[:, j, :J, J:2*J-1] * omega_zeta_factor * sigma_omega_zeta_cross, axis=[1, 2])
        mu_q_ta = mu_q_ta.write(j, (drift + diffusion) / q[:, j])
    mu_q = tf.transpose(mu_q_ta.stack(), [1, 0])
    
    iota = (q - 1) / psi
    Phi = tf.math.log(tf.maximum(psi * iota + 1, 1e-6)) / psi
    phi = 1.0 / omega
    premium = phi * tf.reduce_sum(sigma_qK**2, axis=2)
    residual = (a - iota) / q + Phi - delta + mu_q + sigma * tf.linalg.diag_part(sigma_q) - r - premium

    # Create PDE mask
    pde_mask = tf.ones_like(residual, dtype=tf.float64)
    boundary_mask = ~is_interior
    if tf.reduce_any(boundary_mask):
        boundary_indices = tf.where(boundary_mask)[:, 0]
        boundary_country_boundary = boundary_country[boundary_mask]
        updates = tf.zeros_like(boundary_indices, dtype=tf.float64)
        boundary_country_boundary = tf.cast(boundary_country_boundary, tf.int64)
        indices = tf.stack([boundary_indices, boundary_country_boundary], axis=1)
        pde_mask = tf.tensor_scatter_nd_update(pde_mask, indices, updates)
    
    # Compute PDE loss
    pde_loss = tf.reduce_sum((residual ** 2) * pde_mask) / tf.reduce_sum(pde_mask)

    # Compute boundary loss
    boundary_N = tf.reduce_sum(tf.cast(boundary_mask, tf.float64))
    if boundary_N > 0:
        q_boundary = q[boundary_mask]
        boundary_country_boundary = boundary_country[boundary_mask]
        q_j_boundary = tf.gather(q_boundary, boundary_country_boundary, batch_dims=1)
        loss_boundary = tf.reduce_sum((q_j_boundary - underline_q) ** 2) / boundary_N
    else:
        loss_boundary = tf.constant(0.0, dtype=tf.float64)

    total_loss = pde_loss + loss_boundary
    return total_loss

# Training step
@tf.function
def train_step(batch):
    Omega_batch, is_interior_batch = batch
    with tf.GradientTape() as tape:
        loss = compute_pde_loss_with_mask(Omega_batch, is_interior_batch)
    gradients = tape.gradient(loss, net.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    return loss, gradients

# Main training function
def train_model(epochs_list, learning_rate_list, dataset, net):
    total_epochs = sum(epochs_list)
    current_epoch = 0
    state_cen = tf.convert_to_tensor([[0.5] * J + [1 / J] * (J - 1) + [-1]], dtype=tf.float64)
    num_batches = (N_total + BATCH_SIZE - 1) // BATCH_SIZE

    with open('multi_c_pde.txt', 'a') as f:
        print("Starting main training...", flush=True)
        f.write("Starting main training...\n")
        for phase, (epochs, learning_rate) in enumerate(zip(epochs_list, learning_rate_list)):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
            for epoch in range(epochs):
                start_time = time.time()
                total_loss = 0.0
                batched_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
                for batch in batched_dataset:
                    loss, gradients = train_step(batch)
                    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
                    total_loss += loss.numpy()
                average_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                current_epoch += 1

                q, sigma_q = compute_q_and_sigma_q(state_cen, net)
                q_np = q.numpy()[0]
                sigma_q_np = sigma_q.numpy()[0]
                result_matrix = np.zeros((J + 1, J))
                result_matrix[0, :] = q_np
                result_matrix[1:, :] = sigma_q_np
                df = pd.DataFrame(result_matrix, 
                                 index=['q'] + [f'sigma_q country i{i+1}' for i in range(J)],
                                 columns=[f'Country j{j+1}' for j in range(J)])

                output_str = (f"Phase {phase + 1}, Epoch {current_epoch}/{total_epochs}, "
                              f"Average Loss: {average_loss:.15e}, Time: {elapsed:.2f} seconds, "
                              f"Learning Rate: {learning_rate}\n{df.to_string()}\n")
                print(output_str, flush=True)
                f.write(output_str)
                f.flush()

    net.save_weights('./ckpt/pde_network_checkpoint')
    print("Main training completed.", flush=True)

# Main program
omega_init = np.random.uniform(0.01, 10.0, size=(N_init, J)).astype(np.float64)
omega_init = np.maximum(omega_init, 0.01)
zeta_init = np.random.uniform(0.01, 0.9999, size=(N_init, J - 1)).astype(np.float64)
zeta_init = np.minimum(np.maximum(zeta_init, 0.01), 0.9999)

initialize = Initialize_PDE_Solver(omega_init, zeta_init)
initialize.train(epochs_list=PRETRAIN_EPOCHS, learning_rate_list=PRETRAIN_LEARNING_RATES, batch_size=BATCH_SIZE)

net = PDE_Network()
net.load_weights('./ckpt/pde_initialize_checkpoint')

dataset_generator = DatasetGenerator(N_interior, N_boundary_per_country, J, underline_omega)
dataset = dataset_generator.create_dataset()

with open('multi_c_pde.txt', 'a') as f:
    f.write("PDE Solving Training Log\n\n")

train_model(EPOCHS_MAIN, LEARNING_RATES_MAIN, dataset, net)