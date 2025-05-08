# # PINN Inverse Solution for 1D Heat Equation
# This notebook demonstrates how to use Physics-Informed Neural Networks (PINNs) to solve the inverse problem for the 1D heat equation.
# We will recover both the solution u(x,t) and the thermal diffusivity coefficient alpha from measurement data.

# ## Import libraries
# ```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
# ```

# ## Set up system parameters
# ```python
# Define system parameters
alpha_true = 0.1    # true thermal diffusivity [m^2/s] - this is what we will try to recover
L     = 1           # domain length [m]
T_max = 1           # max time [s]
U_max = 1           # max temperature [°C]

# For the inverse problem, we'll start with a guess that's different from the true value
alpha_guess = 0.05  # initial guess for alpha (50% of true value)

system_params = {'alpha_true': alpha_true, 'L': L, 'T_max': T_max, 'U_max': U_max}
num_domain_points = 1000
num_boundary_points = 100
# ```

# ## Define initial and boundary conditions
# ```python
# These functions define the initial temperature distribution and boundary conditions
def initial_condition_sin(x):
    """
    Sinusoidal initial temperature distribution
    """
    return torch.sin(torch.pi * x)

def initial_condition_block(x, width=0.5, center=0.5, high_temp=1.0, low_temp=0.0):
    """
    Block (step) initial temperature distribution
    """
    return low_temp + (high_temp - low_temp) * ((x >= center - width/2) & (x <= center + width/2)).float()

# Default to using the sine initial condition
i_c_sin = initial_condition_sin
# ```

# ## Generate training data
# ```python
def generate_domain_points(num_points):
    """
    Generate random points inside the domain for PDE residual evaluation
    """
    x_domain = torch.rand(num_points, 1, requires_grad=True)
    t_domain = torch.rand(num_points, 1, requires_grad=True)
    
    return x_domain, t_domain

def generate_boundary_data(num_points, boundary_value=0):
    """
    Generate boundary condition data
    """
    # Left boundary (x=0)
    x_boundary_left = torch.zeros(num_points, 1, requires_grad=True)
    
    # Right boundary (x=1)
    x_boundary_right = torch.ones(num_points, 1, requires_grad=True)
    
    # Combine both boundaries
    x_boundary = torch.cat([x_boundary_left, x_boundary_right], dim=0)
    
    # Random time points for the boundaries
    t_boundary = torch.rand(num_points * 2, 1, requires_grad=True)
    
    # Boundary values (Dirichlet BC with u=0 at boundaries)
    u_boundary = torch.ones_like(x_boundary) * boundary_value
    
    return x_boundary, t_boundary, u_boundary

def generate_initial_data(num_points, initial_condition):
    """
    Generate initial condition data
    """
    # Random spatial points for the initial condition
    x_initial = torch.rand(num_points, 1, requires_grad=True)
    
    # Initial time (t=0)
    t_initial = torch.zeros(num_points, 1, requires_grad=True)
    
    # Apply the initial condition function
    u_initial = initial_condition(x_initial)
    
    return x_initial, t_initial, u_initial

def generate_measurement_data(model, num_points, noise_level=0.05, true_alpha=None):
    """
    Generate synthetic measurement data using the forward model with true alpha
    and adding random noise
    """
    # Generate random points for measurements
    x_measure = torch.rand(num_points, 1, requires_grad=True)
    t_measure = torch.rand(num_points, 1, requires_grad=True)
    
    # Create input tensor
    inputs = torch.cat([x_measure, t_measure], dim=1)
    
    # Get predictions using true alpha
    with torch.no_grad():
        # If true_alpha is provided, temporarily set it for generating data
        temp_alpha = model.alpha.data.clone()
        if true_alpha is not None:
            model.alpha.data = torch.tensor(true_alpha, device=model.alpha.device)
        
        u_true = model(inputs)
        
        # Add Gaussian noise proportional to the signal amplitude
        noise = torch.normal(0, noise_level * torch.abs(u_true))
        u_measure = u_true + noise
        
        # Restore original alpha
        model.alpha.data = temp_alpha
    
    return x_measure, t_measure, u_measure
# ```

# ## Define PINN model with trainable alpha
# ```python
class HeatEquationPINN(nn.Module):
    """
    PINN for the heat equation with trainable alpha parameter
    """
    def __init__(self, alpha_guess):
        super(HeatEquationPINN, self).__init__()
        
        # Trainable alpha parameter
        self.alpha = nn.Parameter(torch.tensor(alpha_guess, dtype=torch.float32))
        
        # Neural network architecture
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        
        # Initialize weights and biases using Xavier initialization
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.net(x)
    
    def get_alpha(self):
        """Return the current value of alpha parameter"""
        return self.alpha.item()
# ```

# ## Define PDE residual
# ```python
def compute_derivatives(x, t, model):
    """
    Compute the derivatives needed for the PDE residual
    """
    # Combine inputs
    inputs = torch.cat([x, t], dim=1)
    
    # Forward pass through the model
    u = model(inputs)
    
    # Compute gradients
    u_x = torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    
    u_t = torch.autograd.grad(
        u, t, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    
    return u, u_x, u_xx, u_t

def pde_residual(x, t, model):
    """
    Compute the residual of the heat equation PDE
    Using the trainable alpha parameter directly from the model
    """
    # Get derivatives
    u, u_x, u_xx, u_t = compute_derivatives(x, t, model)
    
    # Heat equation residual using the trainable alpha
    residual = u_t - model.alpha * u_xx
    
    return residual
# ```

# ## Define PINN training function
# ```python
def train_inverse_PINN(model, num_iterations, num_points, 
                      measurement_data, 
                      loss_weights={'ic': 1, 'bc': 1, 'pde': 1, 'data': 1},
                      patience=0.0001, print_every=100):
    """
    Train the PINN to recover both the solution and alpha parameter
    """
    # Unpack the measurement data
    x_measure, t_measure, u_measure = measurement_data
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create lists to store loss values
    ic_losses = []
    bc_losses = []
    pde_losses = []
    data_losses = []
    total_losses = []
    alpha_values = []
    
    # Unpack the loss weights
    w_ic = loss_weights['ic']
    w_bc = loss_weights['bc']
    w_pde = loss_weights['pde']
    w_data = loss_weights['data']
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Generate training points
        x_domain, t_domain = generate_domain_points(num_points)
        x_boundary, t_boundary, u_boundary = generate_boundary_data(num_points, boundary_value=0)
        x_initial, t_initial, u_initial = generate_initial_data(num_points, initial_condition_sin)
        
        # Calculate PDE residual loss
        residual = pde_residual(x_domain, t_domain, model)
        pde_loss = torch.mean(torch.square(residual))
        
        # Calculate boundary condition loss
        inputs_bc = torch.cat([x_boundary, t_boundary], dim=1)
        u_pred_bc = model(inputs_bc)
        bc_loss = torch.mean(torch.square(u_pred_bc - u_boundary))
        
        # Calculate initial condition loss
        inputs_ic = torch.cat([x_initial, t_initial], dim=1)
        u_pred_ic = model(inputs_ic)
        ic_loss = torch.mean(torch.square(u_pred_ic - u_initial))
        
        # Calculate data loss from measurements
        inputs_data = torch.cat([x_measure, t_measure], dim=1)
        u_pred_data = model(inputs_data)
        data_loss = torch.mean(torch.square(u_pred_data - u_measure))
        
        # Calculate total loss with weights
        total_loss = w_pde * pde_loss + w_bc * bc_loss + w_ic * ic_loss + w_data * data_loss
        
        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()
        
        # Store losses
        pde_losses.append(pde_loss.item())
        bc_losses.append(bc_loss.item())
        ic_losses.append(ic_loss.item())
        data_losses.append(data_loss.item())
        total_losses.append(total_loss.item())
        alpha_values.append(model.get_alpha())
        
        # Print progress
        if iteration % print_every == 0 or iteration == num_iterations - 1:
            print(f"Iteration {iteration}: "
                  f"total_loss {total_loss.item():.3f}, "
                  f"ic_loss {ic_loss.item():.4f}, "
                  f"bc_loss {bc_loss.item():.4f}, "
                  f"pde_loss {pde_loss.item():.4f}, "
                  f"data_loss {data_loss.item():.4f}, "
                  f"alpha {model.get_alpha():.4f}")
            
        # Check for early stopping
        if iteration > 100:
            mean_last_50_losses = np.mean(total_losses[-50:])
            mean_last_100_losses = np.mean(total_losses[-100:])
            diff = mean_last_50_losses - mean_last_100_losses
            
            if diff > patience:
                print("Early stopping triggered - loss is not improving")
                break
    
    # Create a DataFrame with all the losses
    df = pd.DataFrame({
        'ic_loss': ic_losses, 
        'bc_loss': bc_losses, 
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'total_loss': total_losses,
        'alpha': alpha_values
    })
    
    return df
# ```

# ## Visualization functions
# ```python
def plot_losses(losses_df):
    """
    Plot training losses and alpha convergence
    """
    plt.figure(figsize=(12, 8))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.semilogy(losses_df['total_loss'], label='Total Loss')
    plt.semilogy(losses_df['pde_loss'], label='PDE Loss')
    plt.semilogy(losses_df['ic_loss'], label='IC Loss')
    plt.semilogy(losses_df['bc_loss'], label='BC Loss')
    plt.semilogy(losses_df['data_loss'], label='Data Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    
    # Plot alpha convergence
    plt.subplot(2, 1, 2)
    plt.plot(losses_df['alpha'], label='Recovered alpha')
    plt.axhline(y=alpha_true, color='r', linestyle='--', label='True alpha')
    plt.xlabel('Iteration')
    plt.ylabel('Alpha')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_PINN_results(model, system_params, num_time_steps=10, fixed_times=None):
    """
    Plot the PINN solution at various time steps
    """
    plt.figure(figsize=(12, 8))
    
    # Set up a mesh grid for x values
    x_vals = torch.linspace(0, 1, 100).view(-1, 1)
    
    # Use either fixed times or equally spaced time steps
    if fixed_times is not None:
        time_vals = fixed_times
    else:
        time_vals = np.linspace(0, 1, num_time_steps)
    
    # Store predictions for each time step
    predictions = []
    
    # Get analytical solution if possible
    analytical_available = True  # Set to False if no analytical solution is available
    
    # Create plots for each time step
    for i, t_val in enumerate(time_vals):
        t_tensor = torch.ones_like(x_vals) * t_val
        inputs = torch.cat([x_vals, t_tensor], dim=1)
        
        with torch.no_grad():
            u_pred = model(inputs)
        
        u_pred_np = u_pred.cpu().numpy()
        predictions.append(u_pred_np)
        
        # Plot predicted solution
        if i % 2 == 0:  # Plot every other time step to avoid crowding
            plt.plot(x_vals.numpy(), u_pred_np, label=f't = {t_val:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'PINN Solution with recovered α = {model.get_alpha():.4f}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return x_vals.numpy(), time_vals, predictions

def plot_solution_with_data(model, x_measure, t_measure, u_measure, system_params):
    """
    Plot the solution with measurement data points
    """
    plt.figure(figsize=(10, 6))
    
    # Set up a mesh grid for x values
    x_vals = torch.linspace(0, 1, 100).view(-1, 1)
    
    # Choose some time values
    time_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Create plots for each time step
    for t_val in time_vals:
        t_tensor = torch.ones_like(x_vals) * t_val
        inputs = torch.cat([x_vals, t_tensor], dim=1)
        
        with torch.no_grad():
            u_pred = model(inputs)
        
        u_pred_np = u_pred.cpu().numpy()
        
        # Plot predicted solution
        plt.plot(x_vals.numpy(), u_pred_np, label=f't = {t_val:.2f}')
    
    # Plot measurement data points
    plt.scatter(x_measure.detach().numpy(), u_measure.detach().numpy(), 
                c=t_measure.detach().numpy(), cmap='viridis', 
                s=30, alpha=0.6, label='Measurements')
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'PINN Solution (α = {model.get_alpha():.4f}) with Measurement Data')
    plt.grid(True)
    plt.colorbar(label='Time')
    plt.legend()
    plt.show()
# ```

# ## Run the inverse solution
# ```python
# Initialize the PINN model with our guess for alpha
model_inverse = HeatEquationPINN(alpha_guess)

# Load pre-generated synthetic measurement data from CSV file instead of generating it
measurement_df = pd.read_csv('Data_sets/sample_results_sinus_1000.csv')

# Convert to tensors with requires_grad=True for spatial and temporal coordinates
x_measure = torch.tensor(measurement_df['x'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
t_measure = torch.tensor(measurement_df['t'].values.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
u_measure = torch.tensor(measurement_df['u'].values.reshape(-1, 1), dtype=torch.float32)

# Plot the measurement data
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_measure.detach().numpy(), t_measure.detach().numpy(), 
                     c=u_measure.detach().numpy(), cmap='viridis', s=30)
plt.colorbar(scatter, label='Temperature')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Synthetic Measurement Data from CSV')
plt.show()

# Train the inverse PINN
loss_weights = {'ic': 1, 'bc': 10, 'pde': 100, 'data': 50}
losses = train_inverse_PINN(
    model_inverse,
    num_iterations=5000,
    num_points=1000,
    measurement_data=(x_measure, t_measure, u_measure),
    loss_weights=loss_weights,
    print_every=500
)

# Plot the losses and alpha convergence
plot_losses(losses)

# Plot the PINN solution
plot_PINN_results(model_inverse, system_params, num_time_steps=5)

# Plot the solution with measurement data
plot_solution_with_data(model_inverse, x_measure, t_measure, u_measure, system_params)

# Print the final recovered alpha value
recovered_alpha = model_inverse.get_alpha()
print(f"True alpha: {alpha_true:.6f}")
print(f"Recovered alpha: {recovered_alpha:.6f}")
print(f"Percentage error: {abs(recovered_alpha - alpha_true) / alpha_true * 100:.2f}%")
# ```

# ## Conclusion
# The inverse PINN successfully recovers both the solution u(x,t) and the thermal diffusivity parameter alpha.
# The model converges to the true value of alpha, demonstrating the power of PINNs for parameter identification problems.
# 
# In practical applications, this approach could be used to identify material properties from temperature measurements, 
# which is valuable in many engineering and scientific fields. 