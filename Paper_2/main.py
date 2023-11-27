# Please save files here: /home/varqa/schrodinger/main_v1.py (\\wsl.localhost\Ubuntu\home\varqa\schrodinger)

#######
# Import relevant libraries
#######

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from scipy import sparse
from scipy.sparse.linalg import eigsh

import quantum_functions as qf

plt.style.use(["science", "notebook"])

# Declare logger to track code progress
logger = structlog.get_logger()


def calculate(d, delta_d, delta_t) -> tuple[float, float]:
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
    Hard_Wall = 1.00e-6 # Represents infinite potential where F_Total > F_Max

    # Initial confinement
    L = 1.00e-6  # QZE confinement region (m)
    N = 30  # Number PDE spatial grid points in x and y dimension
    delta_X = d / N  # Spatial grid length (m)

    # Extended confinement
    confinement_ratio = 2
    L_ext = confinement_ratio * d
    N_ext = confinement_ratio * N
    delta_X_ext = L_ext / N_ext

    # Set up PDE grids
    X, Y = np.mgrid[0 : d : N * 1j, 0 : d : N * 1j]
    X_ext, Y_ext = np.mgrid[0 : L_ext : N_ext * 1j, 0 : L_ext : N_ext * 1j]

    # Eigenvector of interest
    selected_eignestate = 0

    # Time step for Crank Nicolson
    num_time_steps = 1  # number of time steps

    ########
    # Solve wavefunction for inital confinement and operator matrices
    ########

    # Create confined Hamiltonian H_L
    V = qf.coulomb_potential(X, Y, q_1, q_2, epsilon) + qf.boundary_potential(
        X, Y, q_1, q_2, d, L, Hard_Wall
    )
    U = qf.create_potential_matrix(V, N)
    T = qf.create_kinetic_matrix(N, delta_X, m_1, m_2)
    H = T + U

    # Solve ground state eigenvectors and eigenvalues of H_{L}
    eigenvalues, eigenvectors = eigsh(H, k=selected_eignestate + 1, which="SM")
    print("Ground state eigenvalue:", eigenvalues[selected_eignestate])

    # Select eigenvector of interest
    eigenvector = eigenvectors[:, selected_eignestate]

    # Compute the norm of the eigenvector
    norm = np.linalg.norm(eigenvector)
    print("Norm of the eigenvector:", norm)

    # Obtain 2D eigenvector from 1D eigenvector
    eigenvector_2D = qf.eigenvector_1D_to_2D(eigenvector, N)

    # Obtain probability within boundary
    prob_boundary = qf.probability_within_boundary(
        q_1, q_2, d, L, eigenvector_2D, delta_X, delta_d
    )
    print("Probability within confined boundary:", prob_boundary)

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
    # Leakage
    #######

    # Create extended confined Hamiltonian H_{confinement_ratio*L}
    V_ext = qf.coulomb_potential(X_ext, Y_ext, q_1, q_2, epsilon)+ qf.boundary_potential(
        X_ext, Y_ext, q_1, q_2, d, L, Hard_Wall
    )
    U_ext = qf.create_potential_matrix(V_ext, N_ext)
    T_ext = qf.create_kinetic_matrix(N_ext, delta_X_ext, m_1, m_2)
    H_ext = T_ext + U_ext

    # Crank-Nicolson scheme for time evolution on the extended grid
    A = sparse.eye(N_ext**2) - (1j * delta_t / (2 * hbar)) * H_ext
    B = sparse.eye(N_ext**2) + (1j * delta_t / (2 * hbar)) * H_ext

    # Solve the 1D and 2D respresentatations of the state for each time step
    psi_1D_t, psi_2D_t = qf.solve_future_states(
        num_time_steps, initial_state_1D, A, B, N_ext
    )

    # Calculate leakage
    leakage = qf.calculate_leakage(X, Y, q_1, q_2, d, L, psi_2D_t, N, N_ext)

    return prob_boundary, leakage


# Read in r_initial, delta_r, frequency, delta_t
input_file = "inputs.csv"
df = pd.read_csv(input_file)

boundary_probabilities = []
leakage = []
for index, row in df.iterrows():
    logger.info(f"Calculating for row {index}")
    delta_t = 1 / row.frequency
    b, l = calculate(row.d, row.delta_d, delta_t)
    boundary_probabilities.append(b)
    leakage.append(l)

df["boundary_probabilities"] = boundary_probabilities
df["leakage"] = leakage

print(df)
df.to_csv("output.csv", index=False)
