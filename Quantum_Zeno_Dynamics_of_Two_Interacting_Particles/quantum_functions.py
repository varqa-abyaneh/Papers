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

 

plt.style.use(["science", "notebook"])

 

# Declare logger to track code progress

logger = structlog.get_logger()

 

# Universal constants

Coulomb = 8.99e9

hbar = 1.05457182e-34

 

###########

# Potential functions

###########

 

# Coulomb potential function for ion-ion interaction (including regularisation term)

def coulomb_potential(x, y, q_1, q_2, epsilon):

    V = Coulomb * q_1 * q_2 / (np.sqrt((x - y) ** 2) + epsilon)

    return V




##########

# Matrix functions

##########

 

# Create potential energy matrix for Hamiltonian

def create_potential_matrix(V, N):

    U = sparse.diags(V.reshape(N**2), (0))

    return U

 

# Create kinetic energy matrix for Hamiltonian (finite difference approach)

def create_kinetic_matrix(N, delta_X, m_1, m_2):

    diag_1 = np.ones([N]) * -(hbar**2) / (2 * m_1 * delta_X**2)

    diag_2 = np.ones([N]) * -(hbar**2) / (2 * m_2 * delta_X**2)

    diags_1 = np.array([diag_1, -2 * diag_1, diag_1])

    diags_2 = np.array([diag_2, -2 * diag_2, diag_2])

    D_1 = sparse.spdiags(diags_1, np.array([-1, 0, 1]), N, N)

    D_2 = sparse.spdiags(diags_2, np.array([-1, 0, 1]), N, N)

    T = sparse.kronsum(D_1, D_2)

    return T

 

# Reshape eigenvector into 2-D spatial plot

def eigenvector_1D_to_2D(eigenvector, N):

    eigenvector_2D = eigenvector.T.reshape((N, N))

    return eigenvector_2D

 

##########

# Time evolution functions

##########

 

# Solve future states using Crank Nicholson method

def solve_future_states(num_step, initial_state_1D, A, B, N_ext):

    psi_1D_t = []

    psi_2D_t = []

    psi_current = initial_state_1D

    psi_1D_t.append(psi_current)

    psi_2D_t.append(eigenvector_1D_to_2D(psi_1D_t[0], N_ext))

    for step in range(num_step):

        logger.info(f"Inside step {step}")

        psi_next = sparse.linalg.spsolve(A, B @ psi_current)

        psi_1D_t.append(psi_next)

        # Check for nomralisation

        norm = np.linalg.norm(psi_1D_t[step])

        print(f"Norm of Psi_1D_t[{step}]: {norm}")

        psi_2D_t.append(eigenvector_1D_to_2D(psi_next, N_ext))

        psi_current = psi_next

    return np.array(psi_1D_t), np.array(psi_2D_t)




def calculate_leakage(psi_2D_t, N, N_ext):

    # Calculate the offset for the extended grid

    offset = (N_ext - N) // 2

    # Calculate the absolute square of the wavefunction

    psi_2D_t_abs_square = np.abs(psi_2D_t)**2

    # Sum the absolute square inside the initial grid

    sum_inside = np.sum(psi_2D_t_abs_square[offset:offset+N, offset:offset+N])

    # Sum the absolute square outside the initial grid

    sum_outside = np.sum(psi_2D_t_abs_square) - sum_inside

    # Calculate the leakage

    total_sum = sum_inside + sum_outside

    leakage = sum_outside / total_sum

    print(f"Sum inside initial grid: {sum_inside}")

    print(f"Sum outside initial grid: {sum_outside}")

    print(f"Total sum: {total_sum}")

    print(f"Leakage: {leakage * 100:.25f}%")

    return leakage




##########

# Graphics functions

##########

 

# Define position axis labels

def format_ticks(x, pos):

    return f"{x:.0e}"

 

# Display a manually updating time evolution of the 2D state

def graphic_manual_2D_evolve(num_step, psi_2D_t, X_ext, Y_ext, deltaT):

    def format_ticks(value, tick_number):
        return f"{value:.1f}"  # Display only the base value (e.g., 1, 2)

    formatter = FuncFormatter(format_ticks)

    # Scale data by 10^12 for display purposes
    X_scaled = X_ext * 1e12
    Y_scaled = Y_ext * 1e12

    for step in range(num_step):
        plt.figure(figsize=(8, 8))

        plt.pcolormesh(X_scaled, Y_scaled, np.abs(psi_2D_t[step]) ** 2, cmap="nipy_spectral")

        # Set labels with scaled axis description
        plt.xlabel(r"Ion 1 position (pm)", fontdict={'family': 'Times New Roman', 'size': 20})
        plt.ylabel(r"Ion 2 position (pm)", fontdict={'family': 'Times New Roman', 'size': 20})

        # Set title with Times New Roman font
        plt.title(
            f"Wavefunction evolution after {step * deltaT:.2e} seconds",
            fontdict={'family': 'Times New Roman', 'size': 20}
        )

        # Apply custom tick formatting
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        # Adjust tick label size for better readability
        plt.gca().tick_params(axis='x', which='major', labelsize=12)
        plt.gca().tick_params(axis='y', which='major', labelsize=12)

        plt.show()
