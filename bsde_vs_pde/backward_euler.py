import os
import numpy as np
import tensorflow as tf
import time
import pandas as pd

# Set the data type for network parameters
tf.keras.backend.set_floatx('float64')
# Set the GPU device to use for training (0 for GPU, -1 for CPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Ensure the checkpoint directory exists
os.makedirs('./ckpt', exist_ok=True)
RUN_PRETRAIN = False
# Parameter definitions
J = 5  # Number of countries
N_init = 100000  # Number of pre-training samples
underline_omega = 0.2  # Lower bound for omega
underline_q = 1.29  # Reference value for q
delta_t = 0.01  # Time step
BATCH_SIZE = 600  # Batch size
PRETRAIN_EPOCHS = [20, 20]  # List of epochs for pre-training phases
PRETRAIN_LEARNING_RATES = [1e-3, 1e-4]  # Learning rates for pre-training phases
MAIN_EPOCHS = [100, 100, 100]  # List of epochs for main training phases
MAIN_LEARNING_RATES = [1e-4, 1e-5, 1e-6]  # Learning rates for main training phases
rho = 0.03  # Time discount factor
a = 0.1  # Productivity parameter
delta = 0.05  # Depreciation rate
sigma = 0.023  # Volatility parameter
psi = 5  # Investment adjustment parameter
scale=100

# Neural network definition
class BSDE_Network(tf.keras.Model):
    def __init__(self):
        super(BSDE_Network, self).__init__()
        initializer = tf.keras.initializers.GlorotNormal()
        self.shared1 = tf.keras.layers.Dense(128, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros')
        self.shared2 = tf.keras.layers.Dense(128, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros')
        self.xi_tilde_layers = [tf.keras.layers.Dense(32, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros') for _ in range(J)]
        self.r_layer = tf.keras.layers.Dense(32, activation=tf.math.sin, kernel_initializer=initializer, bias_initializer='zeros')
        self.xi_tilde_out = [tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer) for _ in range(J)]
        self.r_out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer)

    def call(self, x):
        x = self.shared1(x)
        x = self.shared2(x)
        xi_tilde = [self.xi_tilde_out[j](self.xi_tilde_layers[j](x)) for j in range(J)]
        xi_tilde = tf.concat(xi_tilde, axis=1)
        r = self.r_out(self.r_layer(x))
        return xi_tilde, r

# Pretraining network
class Initialize_BSDE_Network(tf.keras.Model):
    def __init__(self):
        super(Initialize_BSDE_Network, self).__init__()
        self.net = BSDE_Network()

    def call(self, omega_init_batch, zeta_init_batch):
        inputs = tf.concat([omega_init_batch, zeta_init_batch], axis=1)
        xi_tilde, r = self.net(inputs)
        zeta = zeta_init_batch
        zeta_sum = tf.reduce_sum(zeta, axis=1, keepdims=True)
        zeta_J = 1.0 - zeta_sum
        Xi = tf.reduce_sum(xi_tilde[:, :-1] * zeta, axis=1, keepdims=True) + xi_tilde[:, -1:] * zeta_J + 1e-6
        xi = rho / Xi * xi_tilde
        q = (a * psi + 1) / (psi * xi + 1)
        sample_size = tf.cast(tf.shape(inputs)[0], tf.float64)
        loss = tf.reduce_sum((q - 1.3)**2) / sample_size + tf.reduce_sum((r - 0.03)**2) / sample_size
        return loss

# Pretraining solver
class Initialize_BSDE_Solver(tf.keras.Model):
    def __init__(self, omega_init, zeta_init):
        super(Initialize_BSDE_Solver, self).__init__()
        self.model = Initialize_BSDE_Network()
        self.omega_init = omega_init
        self.zeta_init = zeta_init

    @tf.function
    def grad(self, omega_batch, zeta_batch):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(omega_batch, zeta_batch)
        grad = tape.gradient(loss, self.model.trainable_variables)
        return grad, loss

    def train(self, epochs_list, learning_rate_list, batch_size):
        total_epochs = sum(epochs_list)
        current_epoch = 0
        dataset = tf.data.Dataset.from_tensor_slices((self.omega_init, self.zeta_init)).shuffle(1000).batch(batch_size, drop_remainder=True)

        with open('multi_c_1step.txt', 'a') as f:
            print('Pre-training history:')
            f.write('Pre-training history:\n')
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
                    log_str = f'Phase {phase + 1}, Epoch {current_epoch}/{total_epochs}, Loss: {train_loss:.3e}, Time: {elapsed:.2f} s, Learning Rate: {learning_rate}'
                    print(log_str, flush=True)
                    f.write(log_str + '\n')
                    f.flush()
                    self.model.net.save_weights('./ckpt/bsde_initialize_checkpoint')

# Dataset generator class
class DatasetGenerator:
    def __init__(self, J, underline_omega, delta_t):
        self.J = J
        self.underline_omega = underline_omega
        self.delta_t = delta_t
        self.N_samples_per_country = 10000  # Samples per country
        self.N_boundary_per_country = 2000  # Boundary samples per country
        self.N_interior_per_country = 8000  # Interior samples per country
        self.N_interior = self.N_interior_per_country * J
        self.N_boundary = self.N_boundary_per_country * J
        self.N_total = self.N_samples_per_country * J

    # Generate interior points
    def sample_interior_points(self):
        omega = np.random.uniform(self.underline_omega + 1e-6, 0.8, size=(self.N_interior, self.J))
        alpha = np.ones(self.J) * 50  # Dirichlet parameters, length J, all values 50
        zeta_full = np.random.dirichlet(alpha, size=self.N_interior)
        zeta = zeta_full[:, :self.J-1]  # Take the first J-1 components as zeta
        boundary_country = np.full((self.N_interior, 1), -1)  # -1 indicates interior points
        return np.concatenate([omega, zeta, boundary_country], axis=1)

    def generate_boundary_samples(self):
        """Vectorized boundary sample generator."""
        # ω samples: uniform draw from 0.2 to 0.8
        omega = np.random.uniform(
            0.2, 0.8,
            size=(self.J * self.N_boundary_per_country, self.J)
        )
    
        # Build indices: each country repeated for its boundary samples
        country_indices = np.repeat(
            np.arange(self.J),
            self.N_boundary_per_country
        )
        sample_indices = np.arange(self.J * self.N_boundary_per_country)
        # Set diagonal element (country weight) to underline_omega
        omega[sample_indices, country_indices] = self.underline_omega
    
        # ζ samples: Dirichlet with concentration α = 50 for each country
        alpha = np.ones(self.J) * 50
        zeta_full = np.random.dirichlet(
            alpha,
            size=self.J * self.N_boundary_per_country
        )
        # Keep only first J-1 components
        zeta = zeta_full[:, : self.J - 1]
    
        # Boundary country flag column
        boundary_country = country_indices.reshape(-1, 1)
    
        # Combine features into Omega_boundary
        Omega_boundary = np.concatenate(
            [omega, zeta, boundary_country],
            axis=1
        )
    
        return Omega_boundary

    # Generate Brownian shocks
    def generate_shocks(self):
        z = np.random.normal(0, np.sqrt(self.delta_t), size=(self.N_total, self.J + 1, self.J))
        z[:, 0, :] = 0  # Set the first row to zero
        return z

     def precompute_inverse(self, z):
        """Precompute matrix inverses for [1, z_scaled] blocks."""
        N = z.shape[0]
        # Constant one column: shape (N, J+1, 1)
        ones = np.ones((N, self.J + 1, 1), dtype=np.float64)
        # Scale z for numerical stability
        z_scaled = z * scale
        # Concatenate ones and scaled z along last axis
        matrix = np.concatenate([ones, z_scaled], axis=2)
        # Invert each block
        inv_matrix = np.linalg.inv(matrix)
        return inv_matrix


    def create_dataset(self):
        Omega_interior = self.sample_interior_points()
        Omega_boundary = self.generate_boundary_samples()
        z_all = self.generate_shocks()
        inv_matrix_all = self.precompute_inverse(z_all)
        is_interior = np.concatenate([np.ones(self.N_interior, dtype=bool),
                                    np.zeros(self.N_boundary, dtype=bool)], axis=0)
        Omega_all = np.concatenate([Omega_interior, Omega_boundary], axis=0)
        dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(Omega_all, dtype=tf.float64),
            tf.convert_to_tensor(z_all, dtype=tf.float64),
            tf.convert_to_tensor(inv_matrix_all, dtype=tf.float64),
            tf.convert_to_tensor(is_interior, dtype=tf.bool)
        ))
        dataset = dataset.shuffle(buffer_size=self.N_total, reshuffle_each_iteration=True)
        return dataset

# Compute all dynamic variables
def compute_all_dynamics(Omega, net):
    global sigma, underline_omega, underline_q
    Omega = tf.convert_to_tensor(Omega, dtype=tf.float64)
    N = tf.shape(Omega)[0]
    J = (tf.shape(Omega)[1] + 1) // 2

    omega = Omega[:, :J]
    zeta = Omega[:, J:]
    zeta_sum = tf.reduce_sum(zeta, axis=1, keepdims=True)
    zeta_J = 1.0 - zeta_sum
    zeta_full = tf.concat([zeta, zeta_J], axis=1)
    xi_tilde, r = net(Omega)
    Xi = tf.reduce_sum(xi_tilde * zeta_full, axis=1, keepdims=True) + 1e-6
    xi = rho / Xi * xi_tilde
    q = (a * psi + 1) / (psi * xi + 1)
    phi = 1.0 / omega
    iota = (q - 1.0) / psi

    def compute_grad_and_hess(omega_sample):
        omega_sample = tf.reshape(omega_sample, [-1])
        with tf.GradientTape() as tape:
            tape.watch(omega_sample)
            zeta = omega_sample[J:]
            zeta_sum = tf.reduce_sum(zeta, keepdims=True)
            zeta_J = 1.0 - zeta_sum
            xi_tilde, r = net(tf.expand_dims(omega_sample, axis=0))
            Xi = tf.reduce_sum(xi_tilde[:, :-1] * zeta, axis=1, keepdims=True) + xi_tilde[:, -1:] * zeta_J + 1e-6
            xi = rho / Xi * xi_tilde
            q = (a * psi + 1) / (psi * xi + 1)
        q_grad_sample = tape.jacobian(q, omega_sample)
        q_grad_sample = tf.squeeze(q_grad_sample, axis=0)
        return q_grad_sample

    q_grad = tf.vectorized_map(compute_grad_and_hess, Omega)


    zeta_sum    = tf.reduce_sum(zeta, axis=1, keepdims=True)                  # [N,1]
    zeta_J      = 1.0 - zeta_sum                                              # [N,1]
    zeta_full   = tf.concat([zeta, zeta_J], axis=1)                           # [N,J]

    sum_partial = tf.reduce_sum(
        q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :],
        axis=2
    )                                                                        # [N,J]

    A0 = q_grad[:, :, :J] * (1 - omega[:, tf.newaxis, :])                   # [N,J,J]
    zeta_deriv = q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :]             # [N,J,J-1]
    zeta_deriv_padded = tf.pad(zeta_deriv, [[0,0],[0,0],[0,1]])             # [N,J,J]


    A = A0 + zeta_deriv_padded                                              # [N,J,J]

    A -= tf.einsum('bi,bj->bij', sum_partial, zeta_full)                    # [N,J,J]

    A -= tf.linalg.diag(q)                                                  # [N,J,J]


    b = sum_partial[:, :, tf.newaxis] * zeta_full[:, tf.newaxis, :]          # [N,J,J]
    b -= q_grad[:, :, :J] * (1 - omega[:, tf.newaxis, :])                   # [N,J,J]

    correction = q_grad[:, :, J:2*J-1] * zeta[:, tf.newaxis, :]              # [N,J,J-1]
    b = tf.concat([
    b[:, :, :-1] - correction,
    b[:, :, -1:]
    ], axis=2)                                                                # [N,J,J]

    sigma_q_intermediate = tf.linalg.solve(A, sigma * b)  # [N, i, k]
    sigma_q = tf.transpose(sigma_q_intermediate, perm=[0, 2, 1]) #[N,J,J]
    sigma_qK = sigma_q + sigma * tf.eye(J, batch_shape=[N], dtype=tf.float64)
    sigma_H = tf.reduce_sum(zeta_full[:, :, tf.newaxis] * sigma_qK, axis=1)

    Phi = tf.math.log(tf.maximum(psi * iota + 1, 1e-6)) / psi
    premium = phi * tf.reduce_sum(sigma_qK**2, axis=2)
    mu_q = r + premium - (a - iota) / q - Phi + delta - sigma * tf.linalg.diag_part(sigma_q)

    sigma_omega = (phi - 1)[:, :, tf.newaxis] * (sigma * tf.eye(J, batch_shape=[N], dtype=tf.float64) + sigma_q)
    sigma_zeta = sigma_qK[:, :J-1, :] - sigma_H[:, tf.newaxis, :]

    R_j = (a - iota) / q + Phi - delta + mu_q + sigma * tf.linalg.diag_part(sigma_q)
    mu_omega = (a - iota) / q + (phi - 1) * (R_j - r - tf.reduce_sum(sigma_qK**2, axis=2)) - rho
    mu_zeta = (premium + r - (a * psi + 1) / (psi * q) + 1.0 / psi)[:, :J-1] - \
              (tf.reduce_sum(zeta_full * (premium + r - (a * psi + 1) / (psi * q) + 1.0 / psi), axis=1, keepdims=True)) - \
              tf.reduce_sum(sigma_H[:, tf.newaxis, :] * sigma_zeta, axis=2)

    return q, phi, iota, r, sigma_q, mu_q, mu_omega, mu_zeta, sigma_omega, sigma_zeta


def update_omega_zeta(omega, zeta, mu_omega, mu_zeta, sigma_omega, sigma_zeta, z):
    #
    batch_size = tf.shape(omega)[0]

    #
    # omega: [batch_size, J] -> [batch_size, 1, J]
    # zeta: [batch_size, J-1] -> [batch_size, 1, J-1]
    omega_expanded = omega[:, tf.newaxis, :]
    zeta_expanded = zeta[:, tf.newaxis, :]

    # mu
    # mu_omega: [batch_size, J] -> [batch_size, 1, J]
    # mu_zeta: [batch_size, J-1] -> [batch_size, 1, J-1]
    mu_omega_expanded = mu_omega[:, tf.newaxis, :]
    mu_zeta_expanded = mu_zeta[:, tf.newaxis, :]

    # sigma * z
    # sigma_omega: [batch_size, J, J], z: [batch_size, J+1, J]
    # -> sigma_omega_z: [batch_size, J+1, J]
    sigma_omega_z = tf.einsum('bij,bkj->bki', sigma_omega, z)

    # sigma_zeta: [batch_size, J-1, J], z: [batch_size, J+1, J]
    # -> sigma_zeta_z: [batch_size, J+1, J-1]
    sigma_zeta_z = tf.einsum('bij,bkj->bki', sigma_zeta, z)


    omega_all = omega_expanded * (1.0 + mu_omega_expanded * delta_t + sigma_omega_z)
    zeta_all = zeta_expanded * (1.0 + mu_zeta_expanded * delta_t + sigma_zeta_z)

    #omega_i_list, zeta_i_list = vectorized_to_lists(omega_all, zeta_all)


    return omega_all, zeta_all   # B * J+1 * J


def compute_bsde_loss(Omega_batch, z_batch, inv_matrix_batch, is_interior_batch, net):

    Omega_batch = tf.convert_to_tensor(Omega_batch, dtype=tf.float64)
    boundary_country_batch = tf.cast(Omega_batch[:, -1], tf.int32)
    Omega_batch = Omega_batch[:, :-1]
    batch_size = tf.shape(Omega_batch)[0]


    q_batch, _, _, r, sigma_q_batch, mu_q_batch, mu_omega_batch, mu_zeta_batch, sigma_omega_batch, sigma_zeta_batch = compute_all_dynamics(
        Omega_batch, net)

    omega = Omega_batch[:, :J]
    zeta = Omega_batch[:, J:]


    omega_all, zeta_all = update_omega_zeta(
        omega, zeta, mu_omega_batch, mu_zeta_batch,
        sigma_omega_batch, sigma_zeta_batch, z_batch)


    # omega_all: [batch_size, J+1, J], zeta_all: [batch_size, J+1, J-1]
    omega_flat = tf.reshape(omega_all, [-1, J])  # [batch_size*(J+1), J]
    zeta_flat = tf.reshape(zeta_all, [-1, J - 1])  # [batch_size*(J+1), J-1]


    Omega_flat = tf.concat([omega_flat, zeta_flat], axis=1)  # [batch_size*(J+1), 2*J-1]
    xi_tilde_flat, _ = net(Omega_flat)  # [batch_size*(J+1), J]


    zeta_sum_flat = tf.reduce_sum(zeta_flat, axis=1, keepdims=True)  # [batch_size*(J+1), 1]
    zeta_J_flat = 1.0 - zeta_sum_flat  # [batch_size*(J+1), 1]
    zeta_full_flat = tf.concat([zeta_flat, zeta_J_flat], axis=1)  # [batch_size*(J+1), J]


    Xi_flat = tf.reduce_sum(xi_tilde_flat * zeta_full_flat, axis=1, keepdims=True) + 1e-6  # [batch_size*(J+1), 1]
    xi_flat = rho / Xi_flat * xi_tilde_flat  # [batch_size*(J+1), J]
    q_flat = (a * psi + 1) / (psi * xi_flat + 1)  # [batch_size*(J+1), J]


    q_all = tf.reshape(q_flat, [batch_size, J + 1, J])  # [batch_size, J+1, J]


    q_batch_expanded = tf.expand_dims(q_batch, axis=1)  # [batch_size, 1, J]
    mu_q_batch_expanded = tf.expand_dims(mu_q_batch, axis=1)  # [batch_size, 1, J]


    q_diff = q_all - q_batch_expanded * mu_q_batch_expanded * delta_t  # [batch_size, J+1, J]

   
    lhs = tf.einsum('bij,bjk->bik', inv_matrix_batch, q_diff)  # [batch_size, J+1, J]

    
    hat_q = lhs[:, 0:1, :]  # [batch_size, 1, J]

    
    beta_scaled = lhs[:, 1:, :]  # [batch_size, J, J]

    
    hat_sigma_q_im = scale * beta_scaled / (hat_q + 1e-6)  # [batch_size, J, J]

    
    hat_sigma_q = tf.transpose(hat_sigma_q_im, perm=[0, 2, 1])


    hat_q = tf.squeeze(hat_q, axis=1)  # [batch_size, J]


    total_loss_per_sample = tf.zeros([batch_size], dtype=tf.float64)


    interior_mask = is_interior_batch  # [batch_size]

    q_diff_loss = (q_batch - hat_q) ** 2  # [batch_size, J]
    sigma_q_diff_loss = (sigma_q_batch - hat_sigma_q) ** 2  # [batch_size, J, J]
    sigma_q_diff_mean = tf.reduce_mean(sigma_q_diff_loss, axis=2)  # [batch_size, J]

    interior_loss_per_country = q_diff_loss + sigma_q_diff_mean  # [batch_size, J]
    interior_loss = tf.reduce_sum(interior_loss_per_country, axis=1)  # [batch_size]


    interior_loss = tf.where(interior_mask, interior_loss, 0.0)
    total_loss_per_sample += interior_loss


    boundary_mask = ~is_interior_batch  # [batch_size]


    boundary_country_one_hot = tf.one_hot(boundary_country_batch, depth=J, dtype=tf.float64)  # [batch_size, J]
    q_j_boundary = tf.reduce_sum(q_batch * boundary_country_one_hot, axis=1)  # [batch_size]
    boundary_loss_j = (q_j_boundary - underline_q) ** 2  # [batch_size]


    exclude_boundary_mask = 1.0 - boundary_country_one_hot  # [batch_size, J]
    interior_loss_other = tf.reduce_sum(interior_loss_per_country * exclude_boundary_mask, axis=1)  # [batch_size]

    total_loss_boundary = boundary_loss_j + interior_loss_other  # [batch_size]


    boundary_loss_contribution = tf.where(boundary_mask, total_loss_boundary, 0.0)
    total_loss_per_sample += boundary_loss_contribution

    total_loss = tf.reduce_mean(total_loss_per_sample)
    return total_loss



# Training step
@tf.function
def bsde_train_step(batch, net):
    Omega_batch, z_batch, inv_matrix_batch, is_interior_batch = batch
    with tf.GradientTape() as tape:
        loss = compute_bsde_loss(Omega_batch, z_batch, inv_matrix_batch, is_interior_batch, net)
    gradients = tape.gradient(loss, net.trainable_variables)
    gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    return loss, gradients

# Compute q and sigma_q
def compute_q_and_sigma_q(Omega, net):
    q, _, _, _, sigma_q, _, _, _, _, _ = compute_all_dynamics(Omega, net)
    return q, sigma_q

# Main training function
def train_bsde_model(epochs_list, learning_rate_list, dataset, net):
    total_epochs = sum(epochs_list)
    current_epoch = 0
    state_cen = tf.convert_to_tensor([[0.5] * J + [1.0 / J] * (J - 1)], dtype=tf.float64)

    num_batches = (dataset_generator.N_total + BATCH_SIZE - 1) // BATCH_SIZE

    with open('multi_c_1step.txt', 'a') as f:
        print("Starting main training...", flush=True)
        f.write("Starting main training...\n")
        for phase, (epochs, learning_rate) in enumerate(zip(epochs_list, learning_rate_list)):
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-8)
            for epoch in range(epochs):
                start_time = time.time()
                total_loss = 0.0
                batched_dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
                for batch in batched_dataset:
                    loss, gradients = bsde_train_step(batch, net)
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
                df = pd.DataFrame(result_matrix, index=['q'] + [f'sigma_q country i{i+1}' for i in range(J)],
                                columns=[f'country j{j+1}' for j in range(J)])

                output_str = (f"Phase {phase + 1}, Epoch {current_epoch}/{total_epochs}, Average Loss: {average_loss:.15e}, Time: {elapsed:.2f} s, Learning Rate: {learning_rate}\n{df.to_string()}\n")
                print(output_str, flush=True)

                f.write(output_str)
                f.flush()

    net.save_weights('./ckpt/bsde_network_checkpoint')
    print("Main training completed.", flush=True)

if RUN_PRETRAIN:
    omega_init = np.random.uniform(0.01, 10.0, size=(N_init, J))
    omega_init = np.maximum(omega_init, 0.01)
    zeta_init  = np.random.uniform(0.01, 0.9999, size=(N_init, J - 1))
    zeta_init  = np.clip(zeta_init, 0.01, 0.9999)

    initialize = Initialize_BSDE_Solver(omega_init, zeta_init)
    initialize.train(
        epochs_list         = PRETRAIN_EPOCHS,
        learning_rate_list  = PRETRAIN_LEARNING_RATES,
        batch_size          = BATCH_SIZE,
    )

    
    initialize.model.net.save_weights(CKPT_INIT)


net = BSDE_Network()


if RUN_PRETRAIN:
    net.load_weights(CKPT_INIT)
    print("Loaded pre-trained weights.")
else:
    print("No pre-training: using random initialization.")

# Load pre-trained weights and perform main training
net = BSDE_Network()
#net.load_weights('./ckpt/bsde_initialize_checkpoint')

# Create dataset
dataset_generator = DatasetGenerator(J, underline_omega, delta_t)
dataset = dataset_generator.create_dataset()

with open('multi_c_1step.txt', 'a') as f:
    f.write("BSDE solution training log\n\n")

train_bsde_model(MAIN_EPOCHS, MAIN_LEARNING_RATES, dataset, net)
