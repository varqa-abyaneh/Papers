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
Hard_Wall = 1.00e-10 # Represents infinite potential where F_Total > F_Max


# Initial confinement
L = 1.00e-6  # Charged plat enclosure (m)
d = 1.00e-12  # QZE confinement region (m)
N = 300  # Number PDE spatial grid points in x and y dimension
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


# Eigenvector of interest
selected_eignestate = 0


# Time step for Crank Nicolson
num_time_steps = 1  # number of time steps
deltaT = 1e-12  # Time step 


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
eigenvalues, eigenvectors = eigsh(H, k=selected_eignestate + 1, which="SM")
print("Energy eigenvalue:", eigenvalues[selected_eignestate])


# Select eigenvector of interest
eigenvector = eigenvectors[:, selected_eignestate]


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
    X_ext, Y_ext, q_1, q_2, confinement_ratio*d, L, Hard_Wall
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

