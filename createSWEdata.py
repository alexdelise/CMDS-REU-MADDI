"""Script that solves that solves the 2D shallow water equations using finite
differences where the momentum equations are taken to be linear, but the
continuity equation is solved in its nonlinear form. The model supports turning
on/off various terms, but in its mst complete form, the model solves the following
set of eqations:

    du/dt - fv = -g*d(eta)/dx + tau_x/(rho_0*H)- kappa*u
    dv/dt + fu = -g*d(eta)/dy + tau_y/(rho_0*H)- kappa*v
    d(eta)/dt + d((eta + H)*u)/dx + d((eta + H)*u)/dy = sigma - w

where f = f_0 + beta*y can be the full latitude varying coriolis parameter.
For the momentum equations, an ordinary forward-in-time centered-in-space
scheme is used. However, the coriolis terms is not so trivial, and thus, one
first finds a predictor for u, v and then a corrected value is computed in
order to include the coriolis terms. In the continuity equation, it's used a
forward difference for the time derivative and an upwind scheme for the non-
linear terms. The model is stable under the CFL condition of

    dt <= min(dx, dy)/sqrt(g*H)    and    alpha << 1 (if coriolis is used)

where dx, dy is the grid spacing in the x- and y-direction respectively, g is
the acceleration of gravity and H is the resting depth of the fluid."""

import time
import numpy as np
import matplotlib.pyplot as plt

# import viz_tools
from matplotlib.colors import LightSource
import os
import random
import torch

# ==================================================================================
# ================================ Parameter stuff =================================
# ==================================================================================
# --------------- Physical parameters ---------------
L_x = 1e6  # Length of domain in x-direction
L_y = 1e6  # Length of domain in y-direction
g = 9.81  # Acceleration of gravity [m/s^2]
H = 100  # Depth of fluid [m]
f_0 = 1e-4  # Fixed part of coriolis parameter [1/s]
beta = 2e-11  # gradient of coriolis parameter [1/ms]
rho_0 = 1024.0  # Density of fluid [kg/m^3)]
tau_0 = 0.1  # Amplitude of wind stress [kg/ms^2]
use_coriolis = True  # True if you want coriolis force
use_friction = False  # True if you want bottom friction
use_wind = False  # True if you want wind stress
use_beta = True  # True if you want variation in coriolis
use_source = False  # True if you want mass source into the domain
use_sink = False  # True if you want mass sink out of the domain
param_string = "\n================================================================"
param_string += "\nuse_coriolis = {}\nuse_beta = {}".format(use_coriolis, use_beta)
param_string += "\nuse_friction = {}\nuse_wind = {}".format(use_friction, use_wind)
param_string += "\nuse_source = {}\nuse_sink = {}".format(use_source, use_sink)
param_string += "\ng = {:g}\nH = {:g}".format(g, H)

# --------------- Computational prameters ---------------
# Define simulation cases with parameters
simulation_cases = {
    "256": {"N_x": 256, "N_y": 256, "max_time_step": 3000},
    "128": {"N_x": 128, "N_y": 128, "max_time_step": 1500},
    "64": {"N_x": 64, "N_y": 64, "max_time_step": 600},
    "32": {"N_x": 32, "N_y": 32, "max_time_step": 200},
    "16": {"N_x": 16, "N_y": 16, "max_time_step": 100},
}

# Select the case
case = "64"  # Options: "256", "64", "32"

# Validate and set parameters based on the selected case
if case in simulation_cases:
    N_x = simulation_cases[case]["N_x"]
    N_y = simulation_cases[case]["N_y"]
    max_time_step = simulation_cases[case]["max_time_step"]
else:
    raise ValueError(
        f"Invalid case specified. Choose one of {list(simulation_cases.keys())}."
    )

dx = L_x / (N_x - 1)  # Grid spacing in x-direction
dy = L_y / (N_y - 1)  # Grid spacing in y-direction
dt = 0.1 * min(dx, dy) / np.sqrt(g * H)  # Time step (defined from the CFL condition)
time_step = 1  # For counting time loop steps
x = np.linspace(-L_x / 2, L_x / 2, N_x)  # Array with x-points
y = np.linspace(-L_y / 2, L_y / 2, N_y)  # Array with y-points
X, Y = np.meshgrid(x, y)  # Meshgrid for plotting
X = np.transpose(X)  # To get plots right
Y = np.transpose(Y)  # To get plots right
param_string += "\ndx = {:.2f} km\ndy = {:.2f} km\ndt = {:.2f} s".format(dx, dy, dt)

# Define friction array if friction is enabled.
if use_friction is True:
    kappa_0 = 1 / (5 * 24 * 3600)
    kappa = np.ones((N_x, N_y)) * kappa_0
    # kappa[0, :] = kappa_0
    # kappa[-1, :] = kappa_0
    # kappa[:, 0] = kappa_0
    # kappa[:, -1] = kappa_0
    # kappa[:int(N_x/15), :] = 0
    # kappa[int(14*N_x/15)+1:, :] = 0
    # kappa[:, :int(N_y/15)] = 0
    # kappa[:, int(14*N_y/15)+1:] = 0
    # kappa[int(N_x/15):int(2*N_x/15), int(N_y/15):int(14*N_y/15)+1] = 0
    # kappa[int(N_x/15):int(14*N_x/15)+1, int(N_y/15):int(2*N_y/15)] = 0
    # kappa[int(13*N_x/15)+1:int(14*N_x/15)+1, int(N_y/15):int(14*N_y/15)+1] = 0
    # kappa[int(N_x/15):int(14*N_x/15)+1, int(13*N_y/15)+1:int(14*N_y/15)+1] = 0
    param_string += "\nkappa = {:g}\nkappa/beta = {:g} km".format(
        kappa_0, kappa_0 / (beta * 1000)
    )

# Define wind stress arrays if wind is enabled.
if use_wind is True:
    tau_x = -tau_0 * np.cos(np.pi * y / L_y) * 0
    tau_y = np.zeros((1, len(x)))
    param_string += "\ntau_0 = {:g}\nrho_0 = {:g} km".format(tau_0, rho_0)

# Define coriolis array if coriolis is enabled.
if use_coriolis is True:
    if use_beta is True:
        f = f_0 + beta * y  # Varying coriolis parameter
        L_R = np.sqrt(g * H) / f_0  # Rossby deformation radius
        c_R = beta * g * H / f_0**2  # Long Rossby wave speed
    else:
        f = f_0 * np.ones(len(y))  # Constant coriolis parameter

    alpha = dt * f  # Parameter needed for coriolis scheme
    beta_c = alpha**2 / 4  # Parameter needed for coriolis scheme

    param_string += "\nf_0 = {:g}".format(f_0)
    param_string += "\nMax alpha = {:g}\n".format(alpha.max())
    param_string += "\nRossby radius: {:.1f} km".format(L_R / 1000)
    param_string += "\nRossby number: {:g}".format(np.sqrt(g * H) / (f_0 * L_x))
    param_string += "\nLong Rossby wave speed: {:.3f} m/s".format(c_R)
    param_string += "\nLong Rossby transit time: {:.2f} days".format(
        L_x / (c_R * 24 * 3600)
    )
    param_string += (
        "\n================================================================\n"
    )

# Define source array if source is enabled.
if use_source:
    sigma = np.zeros((N_x, N_y))
    sigma = 0.0001 * np.exp(
        -((X - L_x / 2) ** 2 / (2 * (1e5) ** 2) + (Y - L_y / 2) ** 2 / (2 * (1e5) ** 2))
    )

# Define source array if source is enabled.
if use_sink is True:
    w = np.ones((N_x, N_y)) * sigma.sum() / (N_x * N_y)

# Write all parameters out to file.
with open("param_output.txt", "w") as output_file:
    output_file.write(param_string)

print(param_string)  # Also print parameters to screen
# ============================= Parameter stuff done ===============================


# ==================================================================================
# ==================== Allocating arrays and initial conditions ====================
# ==================================================================================
def initialize_conditions(N_x, N_y, X, Y, L_x, L_y, ic_set):
    """
    Initialize the conditions for the shallow water simulation.

    Parameters:
        N_x (int): Number of grid points in the x-direction.
        N_y (int): Number of grid points in the y-direction.
        X (ndarray): Meshgrid array for x-coordinates.
        Y (ndarray): Meshgrid array for y-coordinates.
        L_x (float): Length of the domain in the x-direction.
        L_y (float): Length of the domain in the y-direction.

    Returns:
        dict: A dictionary containing initialized arrays for u, v, eta, and temporary variables.
    """
    # Initialize velocity and surface elevation arrays
    u_n = np.zeros((N_x, N_y))  # To hold u at current time step
    u_np1 = np.zeros((N_x, N_y))  # To hold u at next time step
    v_n = np.zeros((N_x, N_y))  # To hold v at current time step
    v_np1 = np.zeros((N_x, N_y))  # To hold v at next time step
    eta_n = np.zeros((N_x, N_y))  # To hold eta at current time step
    eta_np1 = np.zeros((N_x, N_y))  # To hold eta at next time step

    # Temporary variables (each time step) for upwind scheme in eta equation
    h_e = np.zeros((N_x, N_y))
    h_w = np.zeros((N_x, N_y))
    h_n = np.zeros((N_x, N_y))
    h_s = np.zeros((N_x, N_y))
    uhwe = np.zeros((N_x, N_y))
    vhns = np.zeros((N_x, N_y))

    # Initial conditions for u and v
    u_n[:, :] = ic_set[0]  # Initial condition for u
    v_n[:, :] = ic_set[1]  # Initial condition for v
    u_n[-1, :] = 0.0  # Ensuring initial u satisfy BC
    v_n[:, -1] = 0.0  # Ensuring initial v satisfy BC
    
    eta_n = ic_set[2]
    # Initial condition for eta
    # eta_n = np.exp(
    #    -(
    #         (X - np.random.uniform(-L_x / 3, L_x / 3)) ** 2 / (2 * (0.07e6) ** 2)
    #         + (Y - np.random.uniform(-L_y / 4, L_y / 4)) ** 2 / (2 * (0.05e6) ** 2)
    #     )
    # ) + 0.5 * np.exp(
    #     -(
    #         (X - np.random.uniform(-L_x / 5, L_x / 5)) ** 2 / (2 * (0.03e6) ** 2)
    #         + (Y - np.random.uniform(-L_y / 6, L_y / 6)) ** 2 / (2 * (0.04e6) ** 2)
    #     )
    # )

#    eta_n = np.exp(
 #       -(
  #          (X - np.random.uniform(-L_x / 2, L_x / 2)) ** 2 / (2 * (0.05e6) ** 2)
   #         + (Y - np.random.uniform(-L_y / 2, L_y / 2)) ** 2 / (2 * (0.05e6) ** 2)
   #     )
   # )

    return {
        "u_n": u_n,
        "u_np1": u_np1,
        "v_n": v_n,
        "v_np1": v_np1,
        "eta_n": eta_n,
        "eta_np1": eta_np1,
        "h_e": h_e,
        "h_w": h_w,
        "h_n": h_n,
        "h_s": h_s,
        "uhwe": uhwe,
        "vhns": vhns,
    }


def run_shallow_water_simulation(
    max_time_step=5000,
    anim_interval=20,
    sample_interval=1000,
    use_friction=False,
    use_wind=False,
    use_coriolis=True,
    use_source=False,
    use_sink=False,
):
    # Sampling variables.
    eta_list = list()
    u_list = list()
    v_list = list()  # Lists to contain eta and u,v for animation
    time_list = list()  # List to contain time for animation
    hm_sample = list()
    ts_sample = list()
    t_sample = list()  # Lists for Hovmuller and time series
    hm_sample.append(eta_n[:, int(N_y / 2)])  # Sample initial eta in middle of domain
    ts_sample.append(
        eta_n[int(N_x / 2), int(N_y / 2)]
    )  # Sample initial eta at center of domain
    t_sample.append(0.0)  # Add initial time to t-samples

    t_0 = time.perf_counter()  # For timing the computation loop
    time_step = 1

    # # Plot eta_n
    # plt.figure(figsize=(10, 8))
    # plt.contourf(X / 1000, Y / 1000, eta_n, levels=100, cmap="viridis")
    # plt.colorbar(label="Surface Elevation (eta_n)")
    # plt.title("Initial Surface Elevation (eta_n)")
    # plt.xlabel("X (km)")
    # plt.ylabel("Y (km)")
    # plt.show()

    # Initialize PyTorch tensors for recording u, v, eta, and time_step
    eta_tensor = torch.zeros((max_time_step, N_x, N_y), dtype=torch.float32)
    u_tensor = torch.zeros((max_time_step, N_x, N_y), dtype=torch.float32)
    v_tensor = torch.zeros((max_time_step, N_x, N_y), dtype=torch.float32)
    time_steps_tensor = torch.zeros(max_time_step, dtype=torch.float32)

    # Record the initial state
    eta_tensor[0] = torch.tensor(eta_n, dtype=torch.float32)
    u_tensor[0] = torch.tensor(u_n, dtype=torch.float32)
    v_tensor[0] = torch.tensor(v_n, dtype=torch.float32)
    time_steps_tensor[0] = 0.0

    # Main time loop for simulation
    while time_step < max_time_step:
        # Computing values for u and v at next time step
        u_np1[:-1, :] = u_n[:-1, :] - g * dt / dx * (eta_n[1:, :] - eta_n[:-1, :])
        v_np1[:, :-1] = v_n[:, :-1] - g * dt / dy * (eta_n[:, 1:] - eta_n[:, :-1])

        # Add friction if enabled
        if use_friction:
            u_np1[:-1, :] -= dt * kappa[:-1, :] * u_n[:-1, :]
            v_np1[:-1, :] -= dt * kappa[:-1, :] * v_n[:-1, :]

        # Add wind stress if enabled
        if use_wind:
            u_np1[:-1, :] += dt * tau_x[:] / (rho_0 * H)
            v_np1[:-1, :] += dt * tau_y[:] / (rho_0 * H)

        # Use a corrector method to add coriolis if it's enabled
        if use_coriolis:
            u_np1[:, :] = (u_np1[:, :] - beta_c * u_n[:, :] + alpha * v_n[:, :]) / (
                1 + beta_c
            )
            v_np1[:, :] = (v_np1[:, :] - beta_c * v_n[:, :] - alpha * u_n[:, :]) / (
                1 + beta_c
            )

        v_np1[:, -1] = 0.0  # Northern boundary condition
        u_np1[-1, :] = 0.0  # Eastern boundary condition

        # Computing arrays needed for the upwind scheme in the eta equation
        h_e[:-1, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)
        h_e[-1, :] = eta_n[-1, :] + H

        h_w[0, :] = eta_n[0, :] + H
        h_w[1:, :] = np.where(u_np1[:-1, :] > 0, eta_n[:-1, :] + H, eta_n[1:, :] + H)

        h_n[:, :-1] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)
        h_n[:, -1] = eta_n[:, -1] + H

        h_s[:, 0] = eta_n[:, 0] + H
        h_s[:, 1:] = np.where(v_np1[:, :-1] > 0, eta_n[:, :-1] + H, eta_n[:, 1:] + H)

        uhwe[0, :] = u_np1[0, :] * h_e[0, :]
        uhwe[1:, :] = u_np1[1:, :] * h_e[1:, :] - u_np1[:-1, :] * h_w[1:, :]

        vhns[:, 0] = v_np1[:, 0] * h_n[:, 0]
        vhns[:, 1:] = v_np1[:, 1:] * h_n[:, 1:] - v_np1[:, :-1] * h_s[:, 1:]

        # Computing eta values at next time step
        eta_np1[:, :] = eta_n[:, :] - dt * (
            uhwe[:, :] / dx + vhns[:, :] / dy
        )  # Without source/sink

        # Add source term if enabled
        if use_source:
            eta_np1[:, :] += dt * sigma

        # Add sink term if enabled
        if use_sink:
            eta_np1[:, :] -= dt * w

        # Update variables for next iteration
        u_n[:, :] = u_np1[:, :]
        v_n[:, :] = v_np1[:, :]
        eta_n[:, :] = eta_np1[:, :]

        time_step += 1

        # Samples for Hovmuller diagram and spectrum every sample_interval time step
        if time_step % sample_interval == 0:
            hm_sample.append(
                eta_n[:, int(N_y / 2)]
            )  # Sample middle of domain for Hovmuller
            ts_sample.append(
                eta_n[int(N_x / 2), int(N_y / 2)]
            )  # Sample center point for spectrum
            t_sample.append(time_step * dt)  # Keep track of sample times

        # Store eta and (u, v) every anim_interval time step for animations
        if time_step % anim_interval == 0:
            # print("Time: \t{:.2f} hours".format(time_step * dt / 3600))
            # print("Step: \t{} / {}".format(time_step, max_time_step))
            # print("Mass: \t{}\n".format(np.sum(eta_n)))
            u_list.append(u_n.copy())
            v_list.append(v_n.copy())
            eta_list.append(eta_n.copy())
            time_list.append(time_step * dt)  # Keep track of time for animation

        # Append current state to PyTorch tensors
        eta_tensor[time_step - 1] = torch.tensor(eta_n, dtype=torch.float32)
        u_tensor[time_step - 1] = torch.tensor(u_n, dtype=torch.float32)
        v_tensor[time_step - 1] = torch.tensor(v_n, dtype=torch.float32)
        time_steps_tensor[time_step - 1] = (time_step - 1) * dt

        # Print shapes for verification
        # print(f"eta_tensor shape: {eta_tensor.shape}")
        # print(f"u_tensor shape: {u_tensor.shape}")
        # print(f"v_tensor shape: {v_tensor.shape}")
        # print(f"time_steps_tensor shape: {time_steps_tensor.shape}")
    # print(
    #     "Main computation loop done!\nExecution time: {:.2f} s".format(
    #         time.perf_counter() - t_0
    #     )
    # )
    # Return tensors for eta, u, v, and time steps along with other outputs
    # return
    return eta_tensor, u_tensor, v_tensor, time_steps_tensor


# ==================================================================================


def process_simulation_data(eta_list, u_list, v_list, time_list, timeIdx1, timeIdx2):
    """
    Processes simulation data by extracting states and converting them into tensors.

    Parameters:
        eta_list (list): List of eta states from the simulation.
        u_list (list): List of u states from the simulation.
        v_list (list): List of v states from the simulation.
        time_list (list): List of time points from the simulation.
        timeIdx1 (list): List of time indices for the first set of samples.
        timeIdx2 (list): List of time indices for the second set of samples.

    Returns:
        tuple: A tuple containing tensors X1, X2, T1, and T2.
    """

    def extract_simulation_state(
        eta_list, u_list, v_list, time_list, timeIdx1, timeIdx2
    ):
        """
        Extracts the eta, u, and v states for specific time indices from the simulation results.

        Parameters:
            eta_list (list): List of eta states from the simulation.
            u_list (list): List of u states from the simulation.
            v_list (list): List of v states from the simulation.
            timeIdx1 (int): The first time index to extract the states for.
            timeIdx2 (int): The second time index to extract the states for.

        Returns:
            tuple: A tuple containing the eta, u, and v states for the specified time indices.
        """
        eta1 = eta_list[timeIdx1].numpy()
        u1 = u_list[timeIdx1].numpy()
        v1 = v_list[timeIdx1].numpy()
        t1 = time_list[timeIdx1].item()
        eta2 = eta_list[timeIdx2].numpy()
        u2 = u_list[timeIdx2].numpy()
        v2 = v_list[timeIdx2].numpy()
        t2 = time_list[timeIdx2].item()
        return eta1, u1, v1, eta2, u2, v2, t1, t2

    X1 = []
    X2 = []
    T1 = []
    T2 = []

    for i in range(len(timeIdx1)):
        eta1, u1, v1, eta2, u2, v2, t1, t2 = extract_simulation_state(
            eta_list, u_list, v_list, time_list, timeIdx1[i], timeIdx2[i]
        )
        # Combine eta1, u1, and v1 into a PyTorch tensor
        state_tensor1 = torch.tensor(np.stack([eta1, u1, v1]), dtype=torch.float32)
        state_tensor2 = torch.tensor(np.stack([eta2, u2, v2]), dtype=torch.float32)

        # Append to the tensors
        X1.append(state_tensor1)
        X2.append(state_tensor2)
        T1.append(t1)
        T2.append(t2)

        # print(f"State tensor 1 shape: {state_tensor1.shape}")
        # print(f"State tensor 2 shape: {state_tensor2.shape}")

    # Convert lists to tensors
    X1 = torch.stack(X1)
    X2 = torch.stack(X2)
    T1 = torch.tensor(T1, dtype=torch.float32)
    T2 = torch.tensor(T2, dtype=torch.float32)

    # print(f"X1 shape: {X1.shape}")
    # print(f"X2 shape: {X2.shape}")
    # print(f"T1 shape: {T1.shape}")
    # print(f"T2 shape: {T2.shape}")

    return X1, X2, T1, T2


# Example usage:


# Call the simulation function
# Run the simulation multiple times
num_batches = 2500  # Number of batches
num_simulations_per_batch = 1**1  # Number of simulations per batch

# define set of 4 different initial conditions for which to run the simulation
N_x, N_y, L_x, L_y = 64, 64, 1e6, 1e6
A = 0.5 # height of sinusoidal wave
x = np.linspace(-L_x / 2, L_x / 2, N_x)
y = np.linspace(-L_y / 2, L_y / 2, N_y)
X, Y = np.meshgrid(x, y, indexing='ij')  # shape (N_x, N_y)
ic_list = ["Gaussian Bump", "2 Gaussian Bumps", "Sinusoidal Wave Pattern", "Flat Conditions"]


for ic_name in ic_list:
    print(f'\nStarting IC {ic_name}')
    x_allBatches = torch.zeros(num_batches * 4, 3, 64, 64)
    ic_allBatches = torch.zeros(num_batches * 4, 3, 64, 64)

    for batch in range(1, num_batches + 1):
        print(f"\nStarting batch {batch} of {num_batches}...")
        ic_set = {
         "Gaussian Bump": [np.zeros((N_x, N_y)), np.zeros((N_x, N_y)), np.exp(-((X - np.random.uniform(-L_x / 2, L_x / 2)) ** 2 / (2 * (0.05e6) ** 2)
        + (Y - np.random.uniform(-L_y / 2, L_y / 2)) ** 2 / (2 * (0.05e6) ** 2)))] , 
        "2 Gaussian Bumps": [np.zeros((N_x, N_y)), np.zeros((N_x, N_y)), np.exp(-((X - np.random.uniform(-L_x / 3, L_x / 3)) ** 2 / (2 * (0.07e6) ** 2)
        + (Y - np.random.uniform(-L_y / 4, L_y / 4)) ** 2 / (2 * (0.05e6) ** 2)) ) + 0.5 * np.exp(-((X - np.random.uniform(-L_x / 5, L_x / 5)) ** 2 / (2 * (0.03e6) ** 2)
        + (Y - np.random.uniform(-L_y / 6, L_y / 6)) ** 2 / (2 * (0.04e6) ** 2)) )] , 
        "Sinusoidal Wave Pattern": [np.zeros((N_x, N_y)), np.zeros((N_x, N_y)), A * np.sin(2 * np.pi * X / L_x)] , 
        "Flat Conditions": [ 1e-3*(np.random.rand(N_x, N_y) - 0.5), 1e-3*(np.random.rand(N_x, N_y) - 0.5), 1e-3*(np.random.rand(N_x, N_y) - 0.5)],
            }
        ic = ic_set[ic_name]

        x1 = []
        x2 = []
        t1 = []
        t2 = []

    #for i in range(num_simulations_per_batch):
     #   print(
      #      f"\nRunning simulation {i + 1} of {num_simulations_per_batch} in batch {batch}..."
       # )
        conditions = initialize_conditions(N_x, N_y, X, Y, L_x, L_y, ic)
        u_n = conditions["u_n"]
        u_np1 = conditions["u_np1"]
        v_n = conditions["v_n"]
        v_np1 = conditions["v_np1"]
        eta_n = conditions["eta_n"]
        eta_np1 = conditions["eta_np1"]
        h_e = conditions["h_e"]
        h_w = conditions["h_w"]
        h_n = conditions["h_n"]
        h_s = conditions["h_s"]
        uhwe = conditions["uhwe"]
        vhns = conditions["vhns"]
       # if i==0:
        #    u_ic = conditions["u_ic"] # save initial conditions and return in a file later on.
        #    v_ic = conditions["v_ic"] 
        #    eta_ic = conditions["eta_ic"]
        eta_tensor, u_tensor, v_tensor, time_tensor = run_shallow_water_simulation(
            max_time_step=max_time_step,
            anim_interval=20,
            sample_interval=10,
            use_friction=use_friction,
            use_wind=use_wind,
            use_coriolis=use_coriolis,
            use_source=use_source,
            use_sink=use_sink,
        )
        nSamples = 10  # Define the number of samples

        # Ensure timeIdx1 and timeIdx2 are pairwise different

        # Choose timeIdx1 randomly, and timeIdx2 as a small offset after timeIdx1 (close in time)
       # print(len(time_tensor))
        timeIdx1 = np.array([max_time_step // 2])  
        #print(timeIdx1.type)
        # timeIdx1 = random.sample(range(len(time_tensor) - 10), nSamples)
        # timeIdx2 can be before or after timeIdx1, but must be different and within bounds
        offset = 5
        timeIdx2 = timeIdx1 + offset
       # for idx in timeIdx1:
            # Choose an offset between -5 and 5, excluding 0
        #    offset = random.choice([i for i in range(-5, 6) if i != 0])
        #   idx2 = idx + offset
            # Ensure idx2 is within valid range
        #    idx2 = max(0, min(idx2, len(time_tensor) - 1))
        #    timeIdx2.append(idx2)
        
        # timeIdx1 = random.sample(range(len(time_tensor) - 10), nSamples)
        # timeIdx2 = random.sample(range(len(time_tensor) - 10), nSamples)

        # timeIdx2 = [
        #     min(idx + np.random.poisson(5) + 1, len(time_tensor) - 1)
        #     for idx in timeIdx1
        # ]
        # timeIdx2 = [idx + np.random.poisson(5) + 1 for idx in timeIdx1]
        # timeIdx2 = [min(idx, len(time_tensor) - 1) for idx in timeIdx2]  # Ensure indices are within bounds
        X1, X2, T1, T2 = process_simulation_data(
            eta_tensor, u_tensor, v_tensor, time_tensor, timeIdx1, timeIdx2
        )
        # Append the current simulation tensors to the batch tensors
        x1.append(X1)
        x2.append(X2)
        t1.append(T1)
        t2.append(T2)

        # Concatenate all tensors in the batch
        x1 = torch.cat(x1, dim=0)
        x2 = torch.cat(x2, dim=0)
        t1 = torch.cat(t1, dim=0).unsqueeze(1)
        t2 = torch.cat(t2, dim=0).unsqueeze(1)
        u_ic_torch = torch.from_numpy(ic[0])
        v_ic_torch = torch.from_numpy(ic[1])
        eta_ic_torch = torch.from_numpy(ic[2])
        ic_torch = torch.cat([eta_ic_torch, u_ic_torch, v_ic_torch], dim=0)
        
        x_allBatches[batch-1, :, :, :] = x1
        ic_allBatches[batch-1, 0, :, :] =  eta_ic_torch
        ic_allBatches[batch-1, 1, :, :] =  u_ic_torch
        ic_allBatches[batch-1, 2, :, :] =  v_ic_torch

        # Save the tensors for the current batch to files
    os.makedirs("new_data", exist_ok=True)
    #print(ic_name)
    torch.save(x_allBatches, f"data/train_x1_{ic_name}_allBatch_Close.pt")
    torch.save(ic_allBatches, f"data/train_ic_{ic_name}_allBatch.pt")
        # torch.save(x2, f"data/x2_{ic_name}_batch{batch}Close.pt")
        # torch.save(t1, f"data/t1_{ic_name}_batch{batch}Close.pt")
        # torch.save(t2, f"data/t2_{ic_name}_batch{batch}Close.pt")
    
        # Plot one example from x1 and x2 with titles t1 and t2

    print(f"Batch {batch} - X1 shape: {x1.shape}")
    print(f"Batch {batch} - X2 shape: {x2.shape}")
    print(f"Batch {batch} - T1 shape: {t1.shape}")
    print(f"Batch {batch} - T2 shape: {t2.shape}")

print("All batches completed and saved successfully.")

print("State and time tensors saved successfully.")


plt.show()

