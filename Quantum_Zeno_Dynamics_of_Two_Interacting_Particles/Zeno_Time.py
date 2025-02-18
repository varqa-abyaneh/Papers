import math
import matplotlib.pyplot as plt
import numpy as np
import quantum_functions as qf
from scipy.sparse.linalg import eigsh

# Universal constants
hbar = 1.05457182e-34

# Mass and charge. Both ions are protons
m_1 = 1.67262192e-27  # 1st particle mass (kg)
m_2 = 1.67262192e-27  # 2nd particle mass (kg)
q_1 = 1.60217663e-19  # 1st particle charge (C)
q_2 = 1.60217663e-19  # 2nd particle charge (C)

# Numerical parameters
epsilon = 1e-15  # Regularisation to stop potential energy singularity (m)
N =300 # Number of PDE spatial grid points

# Extended confinement ratio
confinement_ratio = 3

# Distance values

distances = [1e-12,1e-11,1e-10,1e-9,1e-8,1e-7]

zeno_times = []

# Loop over distances
for d in distances:
    delta_X = d / N  # Spatial grid length (m)
    
    # Extended confinement
    d_ext = confinement_ratio * d
    N_ext = confinement_ratio * N
    
    # Set up PDE grids
    X, Y = np.mgrid[0:d:N*1j, 0:d:N*1j]
    
    # Create confined Hamiltonian H_d
    V = qf.coulomb_potential(X, Y, q_1, q_2, epsilon)
    U = qf.create_potential_matrix(V, N)
    T = qf.create_kinetic_matrix(N, delta_X, m_1, m_2)
    H = T + U
    
    # Solve ground state eigenvectors and eigenvalues
    eigenvalues, eigenvectors = eigsh(H, k=1, which="SM")
    
    # Select the eigenvector of interest
    eigenvector = eigenvectors[:, 0]
    
    # Compute ⟨H⟩
    psi_conjugate = np.conjugate(eigenvector)
    H_psi = H @ eigenvector  # Apply Hamiltonian to the eigenvector
    H_expectation = np.vdot(psi_conjugate, H_psi)  # <psi|H|psi>
    
    # Compute ⟨H²⟩
    H2_psi = H @ H_psi  # Apply Hamiltonian again
    H2_expectation = np.vdot(psi_conjugate, H2_psi)  # <psi|H^2|psi>
    
    # Compute the variance
    variance = H2_expectation - H_expectation**2
    
    # Compute the Zeno time
    zeno_time = hbar / math.sqrt(abs(variance.real))
    zeno_times.append(zeno_time)

# Plot Zeno time vs. distance
plt.figure(figsize=(8, 6))
plt.plot(distances, zeno_times, marker='o', label="Zeno Time")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("$d$ (m)")
plt.ylabel("Zeno Time (s)")
plt.title("Zeno Time vs. Distance")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()