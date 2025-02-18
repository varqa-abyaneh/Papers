#######

# This code calculates the leakage probability
# as a function of time for diffeent choices of 
# "d", "N", "number of steps"..

#######

 

import marshal
import math

import matplotlib.pyplot as plt

import numpy as np

import scienceplots

import structlog

from matplotlib import animation

from matplotlib.animation import FuncAnimation, PillowWriter

from matplotlib.ticker import FuncFormatter

from mpl_toolkits.mplot3d import Axes3D

from scipy import sparse

from scipy.sparse.linalg import eigsh

import quantum_functions as qf


plt.style.use(["science", "notebook"])


# Declare logger to track code progress

logger = structlog.get_logger()

# Universal constants

hbar = 1.05457182e-34

# Mass and charge. Both ions are protons

m_1 = 1.67262192e-27  # 1st particle mass (kg)

m_2 = 1.67262192e-27  # 2st particle mass (kg)

q_1 = 1.60217663e-19  # 1st particle charge (C)

q_2 = 1.60217663e-19  # 2st particle charge (C)

# Numerical parameters

epsilon = 1e-15  # Regularisation to stop potential energy singularity (m)

# Initial confinement

d = 1.00e-12 # QZE confinement region (m)

N = 50  # Number PDE spatial grid points in x and y dimension

delta_X = d / N  # Spatial grid length (m)

distance_threshold = 1e-15  # fusion threshold distance (m)

# Extended confinement. Required for calculating leakage

confinement_ratio = 3

d_ext = confinement_ratio * d

N_ext = confinement_ratio * N

delta_X_ext = d_ext / N_ext

# Set up PDE grids

X, Y = np.mgrid[0 : d : N * 1j, 0 : d : N * 1j]

X_ext, Y_ext = np.mgrid[0 : d_ext : N_ext * 1j, 0 : d_ext : N_ext * 1j]

# Eigenvector of interest

selected_eignestate = 0

# Time step for Crank Nicolson

num_time_steps = 60   # number of time steps

deltaT = 3e-20

########

# Solve wavefunction for inital confinement and operator matrices

########

# Create confined Hamiltonian H_d

V = qf.coulomb_potential(X, Y, q_1, q_2, epsilon)

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


# Time evolution

# Create extended confined Hamiltonian H_{confinement_ratio*L}
V_ext = qf.coulomb_potential(X_ext, Y_ext, q_1, q_2, epsilon)
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

leakage = qf.calculate_leakage(psi_2D_t[-1], N, N_ext)

import os
from scipy.sparse.linalg import spsolve

def calculate_leakage_over_time(num_time_steps, deltaT, initial_state_1D, A, B, N_ext, N, qf):
    """Calculate leakage values over time for a two-ion system."""
    times = np.arange(0, num_time_steps * deltaT, deltaT)
    psi_1D_t = initial_state_1D.copy()
    leakage_values = []

    for _ in times:
        psi_1D_t = spsolve(A, B @ psi_1D_t)
        psi_2D_t = psi_1D_t.reshape((N_ext, N_ext))
        leakage_values.append(qf.calculate_leakage(psi_2D_t, N, N_ext))

    return times, leakage_values

# Calculate data
times, leakage_values = calculate_leakage_over_time(num_time_steps, deltaT, initial_state_1D, A, B, N_ext, N, qf)

# Save data
np.savetxt("leakage_vs_time1.csv", np.column_stack((times, leakage_values)),
           delimiter=",", header="Time (s), Leakage", comments="", fmt="%.8e")

# Plot data
plt.plot(times, leakage_values, 'go-', label="Leakage")
plt.title("Leakage Function vs. Time", fontsize=20, family="Times New Roman")
plt.xlabel("Time (s)", fontsize=20, family="Times New Roman")
plt.ylabel("Leakage Probability", fontsize=20, family="Times New Roman")
plt.xticks(fontsize=16, family="Times New Roman")
plt.yticks(fontsize=16, family="Times New Roman")
plt.grid(True)
plt.tight_layout()
plt.show()

