# Please save files here: /home/varqa/schrodinger/main_v1.py (\\wsl.localhost\Ubuntu\home\varqa\schrodinger)

#######
# Import relevant libraries
#######

import marshal
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import structlog
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
import quantum_functions as qf

plt.style.use(["science", "notebook"])

# Declare logger to track code progress
logger = structlog.get_logger()


########
# Setup
########

# Universal constants
hbar = 1.05457182e-34

# Mass and charge. Both ions are protons 
m_1 = 1.67262192e-27  # 1st particle mass (kg)
m_2 = 1.67262192e-27  # 2st particle mass (kg)
q_1 = 1.60217663e-19  # 1st particle charge (C)
q_2 = 1.60217663e-19  # 2st particle charge (C)

# Numerical parameters
epsilon = 1e-15  # Regularisation to stop potential energy singularity (m)
Hard_Wall = 0 # Represents infinite potential where F_Total > F_Max

# Initial confinement
L = 1.00e-6
d = 1.00e-12  # QZE confinement region (m)
N = 30 # Number PDE spatial grid points in x and y dimension
delta_X = d / N  # Spatial grid length (m)
distance_threshold = 1e-15  # fusion threshold distance (m)


# Extended confinement. Required for calculating leakage
confinement_ratio = 2 
d_ext = confinement_ratio * d
N_ext = confinement_ratio * N
delta_X_ext = d_ext / N_ext

# Set up PDE grids
X, Y = np.mgrid[0 : d : N * 1j, 0 : d : N * 1j]
X_ext, Y_ext = np.mgrid[0 : d_ext : N_ext * 1j, 0 : d_ext : N_ext * 1j]

# Time step for Crank Nicolson
num_time_steps = 1  # number of time steps
deltaT = 5e-15  # Time step (currently 1e-20 seconds seems to be scale to evolve 1e-13m to 2e-13m)


########
# Solve wavefunction for inital confinement and operator matrices
########


# Create confined Hamiltonian H_d
V = qf.coulomb_potential(X, Y, q_1, q_2, epsilon) + qf.boundary_potential(
    X, Y, q_1, q_2, d, L, Hard_Wall
)
U = qf.create_potential_matrix(V, N)
T = qf.create_kinetic_matrix(N, delta_X, m_1, m_2)
H = T + U

# Solve ground state eigenvectors and eigenvalues of H_{L}
eigenvalues, eigenvectors = eigsh(H, k=1, which="SM")
print("Ground state eigenvalue:", eigenvalues[0])

# Select eigenvector of interest
eigenvector = eigenvectors[:, 0]

# Compute the norm of the eigenvector
norm = np.linalg.norm(eigenvector)
print("Norm of the eigenvector:", norm)

# Obtain 2D eigenvector from 1D eigenvector
eigenvector_2D = qf.eigenvector_1D_to_2D(eigenvector, N)

# Calculate the probability particles are within 1e-15m of one another
prob = qf.probability_within_distance(eigenvector_2D, delta_X, distance_threshold)
print(f"The probability of |x1 - x2| < {distance_threshold} m is {prob}")


#######
# Create initial state on extended PDE grid
#######


# initial_psi = np.zeros((N_ext, N_ext), dtype=np.complex128)
initial_state_2D = np.zeros((N_ext, N_ext))
offset = (N_ext - N) // 2
initial_state_2D[offset : offset + N, offset : offset + N] = eigenvector_2D

# Obtain 1D initial state from 2D initial state
initial_state_1D = initial_state_2D.flatten()

# Compute the norm of the eigenvector
norm = np.linalg.norm(initial_state_1D)
print("Norm of the eigenvector:", norm)


#######
# Time evolution
#######


# Create extended confined Hamiltonian H_{confinement_ratio*L}
V_ext = qf.coulomb_potential(X_ext, Y_ext, q_1, q_2, epsilon) + qf.boundary_potential(
    X_ext, Y_ext, q_1, q_2, confinement_ratio*d, L, 1e-3
)
U_ext = qf.create_potential_matrix(V_ext, N_ext)
T_ext = qf.create_kinetic_matrix(N_ext, delta_X_ext, m_1, m_2)
H_ext = T_ext + U_ext


# Crank-Nicolson scheme for time evolution on the extended grid
A = sparse.eye(N_ext**2) - (1j * deltaT / (2 * hbar)) * H_ext
B = sparse.eye(N_ext**2) + (1j * deltaT / (2 * hbar)) * H_ext

# Solve the 1D and 2D respresentatations of the state for each time step
psi_1D_t, psi_2D_t = qf.solve_future_states(
    num_time_steps, initial_state_1D, A, B, N_ext
)

# Calculate leakage
leakage = qf.calculate_leakage(X, Y, q_1, q_2, d, L, psi_2D_t, N, N_ext)


# ######
# Graphics
# ######

# Manual graphical update 2D wave function
# qf.graphic_manual_2D_evolve(num_time_steps, psi_2D_t, X_ext, Y_ext, deltaT)

# 2D animation
# ql.graphic_animation_2D(num_time_steps, psi_2D_t, deltaT, frames_per_second=2, filename = '2D_10e-12.gif')

# Manual graphical update 1D where x axis is |x-y|
# ql.graphic_manual_1D_evolve(psi_2D_t, num_time_steps, deltaT, delta_X_ext)

# 1D animation
# ql.graphic_animation_1D(num_time_steps, psi_2D_t, deltaT, frames_per_second=2, filename = '1D_10e-12.gif')

# Plot eigenvector on initial confinement
formatter = FuncFormatter(qf.format_ticks)
plt.figure(figsize=(8, 8))
plt.pcolormesh(X, Y, eigenvector_2D**2, cmap="nipy_spectral")
plt.xlabel("Particle 1 Position (m)")
plt.ylabel("Particle 2 Position (m)")
plt.axis("on")
plt.title(f"Ground State: deuterium-tritium QZE trapped in {d:.1e}m region")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
# Flatten the X, Y, and squared eigenvector arrays for saving
X_flat = X.flatten()
Y_flat = Y.flatten()
eigenvector_2D_flat = (eigenvector_2D**2).flatten()
# Combine the flattened arrays
combined_array = np.column_stack((X_flat, Y_flat, eigenvector_2D_flat))
# Save to CSV
np.savetxt("/home/varqa/schrodinger/2d_wavefunction_data.csv", combined_array, delimiter=",", header="Particle 1 Position,Particle 2 Position,Eigenvector Squared", comments='')
# Print message to confirm saving
print("Data saved to /home/varqa/schrodinger/2d_wavefunction_data.csv")


# # Plot same eigenvector on extended grid
# plt.figure(figsize=(8,8))
# plt.pcolormesh(X_ext, Y_ext, initial_state_2D**2, cmap='nipy_spectral')
# plt.axis('on')
# plt.show()


# Plot relative_expected_momentum against time
# Create a time array
time_array = np.arange(0, (num_time_steps+1) * deltaT, deltaT)
formatter = FuncFormatter(qf.format_ticks)
plt.figure(figsize=(8,8))



plt.plot(time_array, relative_expected_momentum_t)
plt.title(f"Relative expected momentum against time")
plt.xlabel("Time (s)")  # Assuming delta_T is in seconds
plt.ylabel("Relative expected momentum (kg m/s)")
plt.gca().xaxis.set_major_formatter(formatter)
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()
# Combine the time_array and relative_expected_momentum_t for saving
time_array_real = np.real(time_array)
relative_expected_momentum_t_real = np.real(relative_expected_momentum_t)
combined_array = np.column_stack((time_array_real, relative_expected_momentum_t_real))
# Save to CSV
np.savetxt("relative_expected_momentum_vs_time.csv", combined_array, delimiter=",", header="Time,Relative Expected Momentum", comments='')

# # Plot relative_expected_momentum squared against time
# # Create a time array
# time_array = np.arange(0, (num_time_steps+1) * deltaT, deltaT)

# # Plotting
# formatter = FuncFormatter(ql.format_ticks)
# plt.figure(figsize=(8,8))
# plt.plot(time_array, relative_expected_momentum_squared_t)
# plt.title("Squared Relative Expected Momentum against Time")
# plt.xlabel("Time (s)")  # Assuming deltaT is in seconds
# plt.ylabel("Squared Relative Expected Momentum (kg^2 m^2/s^2)")
# plt.gca().xaxis.set_major_formatter(formatter)
# plt.gca().yaxis.set_major_formatter(formatter)
# plt.show()
# # Convert to real parts if necessary
# time_array_real = np.real(time_array)
# relative_expected_momentum_squared_t_real = np.real(relative_expected_momentum_squared_t)
# # Combine the time array and squared relative momentum for saving
# combined_squared_array = np.column_stack((time_array_real, relative_expected_momentum_squared_t_real))
# # Save to CSV
# np.savetxt("relative_expected_momentum_squared_vs_time.csv", combined_squared_array, delimiter=",", header="Time,Squared Relative Expected Momentum", comments='')
