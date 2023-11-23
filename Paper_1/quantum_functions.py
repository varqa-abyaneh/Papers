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


# This function creates the boundary potential due to the QZE confinement measurements
def boundary_potential(x, y, q_1, q_2, d, L, Hard_Wall):
    x_1 = x + (L / 2 - d / 2)
    x_2 = y + (L / 2 - d / 2)
    F_Total = (
        Coulomb * q_1**2 / (4 * x_1**2)
        + Coulomb * q_2**2 / (4 * x_2**2)
        + Coulomb * q_1**2 / (4 * (L - x_1) ** 2)
        + Coulomb * q_2**2 / (4 * (L - x_2) ** 2)
    )
    F_Max = (
        Coulomb * q_1**2 / ((L - d) ** 2)
        + Coulomb * q_1**2 / ((L + d) ** 2)
        + 2 * Coulomb * q_2**2 / (L**2)
    )
    V = np.where(F_Total < F_Max, 0, Hard_Wall)
    return V


# Define a Gaussian function
def gaussian(x, y, x0, y0, width):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * width**2))


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


# A matrix representing the relative momentum between ion 1 and ion 2
def create_relative_momentum_matrix(N, delta_X):
    # Create the single-particle momentum operator for one dimension
    diag = np.zeros(N)
    off_diag = 0.5j * np.ones(N - 1)
    single_particle_momentum = sparse.diags(
        [off_diag, diag, -off_diag], [-1, 0, 1], shape=(N, N)
    )
    # Create the identity matrix
    I = sparse.eye(N)
    p_rel = sparse.kron(single_particle_momentum, I) - sparse.kron(
        I, single_particle_momentum
    )
    p_rel = p_rel * (hbar / delta_X)
    return p_rel

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


# Obtain time-series of expectation of relative momentum, given time-series of quantum states
def solve_future_expected_relative_momentum(num_step, psi_1D_t, p_rel_ext):
    relative_expected_momentum_t = []
    for step in range(num_step):
        rel_next = -np.dot(np.conj(psi_1D_t[step]), p_rel_ext.dot(psi_1D_t[step]))
        relative_expected_momentum_t.append(rel_next)
    return relative_expected_momentum_t


# Obtain time-series of expectation of square of relative momentum, given time-series of quantum states
def solve_future_expected_relative_momentum_squared(num_step, psi_1D_t, p_rel_ext):
    relative_expected_momentum_squared_t = []
    p_rel_squared = p_rel_ext.dot(p_rel_ext)  # Square the relative momentum operator

    for step in range(num_step):
        psi_step = psi_1D_t[step]
        expectation_value_squared = np.dot(np.conj(psi_step), p_rel_squared.dot(psi_step))
        relative_expected_momentum_squared_t.append(np.abs(expectation_value_squared))

    return np.array(relative_expected_momentum_squared_t)


##########
# Probability functions
##########


# Probability of fusion for a given a quantum state
def probability_within_distance(eigenvector_2D, delta_X, distance_threshold=1e-15):
    N = eigenvector_2D.shape[0]
    prob = 0.0

    for i in range(N):
        for j in range(N):
            x1 = i * delta_X
            x2 = j * delta_X
            if abs(x1 - x2) < distance_threshold:
                prob += np.abs(eigenvector_2D[i, j]) ** 2

    return prob


# Probability we can tighten QZE boundary by delta_d
def probability_within_boundary(
    q_1: float, q_2: float, d: float, L: float, eigenvector_2D, delta_X, delta_d: float
) -> float:
    d = d - (2 * delta_d)
    N = eigenvector_2D.shape[0]
    prob = 0.0

    for i in range(N):
        for j in range(N):
            x_1 = i * delta_X
            x_2 = j * delta_X

            x_1_adj = x_1 + (L / 2 - d / 2)
            x_2_adj = x_2 + (L / 2 - d / 2)

            F_Total = (
                Coulomb * q_1**2 / (4 * x_1_adj**2)
                + Coulomb * q_2**2 / (4 * x_2_adj**2)
                + Coulomb * q_1**2 / (4 * (L - x_1_adj) ** 2)
                + Coulomb * q_2**2 / (4 * (L - x_2_adj) ** 2)
            )
            F_Max = (
                Coulomb * q_1**2 / ((L - d) ** 2)
                + Coulomb * q_1**2 / ((L + d) ** 2)
                + 2 * Coulomb * q_2**2 / (L**2)
            )
            if F_Total < F_Max:
                # logger.debug("Adding probability", x_1=x_1, x_2=x_2)
                prob += np.abs(eigenvector_2D[i, j]) ** 2
    return prob


# Calculate leakage probability 
def calculate_leakage(
    x: float,
    y: float,
    q_1: float,
    q_2: float,
    d: float,
    L: float,
    psi_2D_t,
    N: float,
    N_ext: float,
) -> float:
    x_1 = x + (L / 2 - d / 2)
    x_2 = y + (L / 2 - d / 2)
    
    F_Total = (
        Coulomb * q_1**2 / (4 * x_1**2)
        + Coulomb * q_2**2 / (4 * x_2**2)
        + Coulomb * q_1**2 / (4 * (L - x_1) ** 2)
        + Coulomb * q_2**2 / (4 * (L - x_2) ** 2)
    )
    F_Max = (
        Coulomb * q_1**2 / ((L - d) ** 2)
        + Coulomb * q_1**2 / ((L + d) ** 2)
        + 2 * Coulomb * q_2**2 / (L**2)
    )
    V = np.where(F_Total < F_Max, 1, 0)
    assert V.shape == F_Total.shape, 'Misaligned shapes'

    # Calculate the offset for the extended grid
    offset = (N_ext - N) // 2

    # Insert V into offset position in V_ext
    V_ext = np.zeros((N_ext, N_ext))
    V_ext[offset : offset + N, offset : offset + N] = V

    # Calculate the absolute square of the wavefunction
    psi_2D_t_abs_square_latest = np.abs(psi_2D_t[-1]) ** 2

    # Sum the absolute square inside the initial grid
    assert (
        psi_2D_t_abs_square_latest.shape == V_ext.shape
    ), "Mismatch in shape between psi and V_ext"
    sum_inside = np.sum(psi_2D_t_abs_square_latest[V_ext == 1])

    # Sum the absolute square outside the initial grid
    sum_outside = np.sum(psi_2D_t_abs_square_latest[V_ext == 0])

    # Calculate the leakage
    total_sum = sum_inside + sum_outside
    leakage = sum_outside / total_sum
    print(f"Sum inside initial grid: {sum_inside}")
    print(f"Sum outside initial grid: {sum_outside}")
    print(f"Total sum: {total_sum}")
    print(f"Leakage: {leakage * 100:.4f}%")
    return leakage


##########
# Graphics functions
##########


# Define position axis labels
def format_ticks(x, pos):
    return f"{x:.0e}"


# Display a manually updating time evolution of the 2D state
def graphic_manual_2D_evolve(num_step, psi_2D_t, X_ext, Y_ext, deltaT):
    formatter = FuncFormatter(format_ticks)
    for step in range(num_step):
        plt.figure(figsize=(8, 8))
        plt.pcolormesh(X_ext, Y_ext, np.abs(psi_2D_t[step]) ** 2, cmap="nipy_spectral")
        plt.xlabel("Particle 1 Position (m)")
        plt.ylabel("Particle 2 Position (m)")
        plt.axis("on")
        # plt.title(f"Time evolution following QZE measurment: {step * deltaT:.2e} seconds")
        plt.title(
            f"Time evolution following QZE boundary extension: {step * deltaT:.2e} seconds"
        )
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()


# Display an animation updating time evolution of the 2D state
def graphic_animation_2D(
    num_frames, psi_2D_t, deltaT, frames_per_second, filename=None
):
    formatter = FuncFormatter(format_ticks)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    c = ax.pcolormesh(np.abs(psi_2D_t[0]) ** 2, cmap="nipy_spectral")
    ax.set_xlabel("Particle 1 Position (m)")
    ax.set_ylabel("Particle 2 Position (m)")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    fig.colorbar(c, ax=ax, label="Probability Density")

    # Update function for animation
    def update(frame):
        c.set_array(np.abs(psi_2D_t[frame]) ** 2)
        ax.set_title(f"Time: {frame * deltaT:.2e} seconds")
        return (c,)

    # Create animation
    ani = FuncAnimation(
        fig,
        update,
        frames=range(num_frames),
        repeat=False,
        interval=1000 / frames_per_second,
    )
    # Save animation
    if filename:
        ani.save(filename, writer="ffmpeg", fps=frames_per_second)
    # Show animation
    plt.show()


# Display a manually updating time evolution of the 1D state where x axis is |x-y|
def graphic_manual_1D_evolve(psi_2D_t, steps, deltaT, delta_X_ext):
    formatter = FuncFormatter(format_ticks)
    n = psi_2D_t.shape[1]  # Assuming psi_2D_t has shape (steps, n, n)
    for step in range(steps):
        psi_2D = psi_2D_t[step]
        # Calculate the probability amplitude for each distance |x - y|
        distance_prob = np.zeros(n)
        scaled_distances = np.zeros(n)  # To hold the scaled distances
        for i in range(n):
            for j in range(n):
                distance = abs(i - j)
                scaled_distance = delta_X_ext * distance  # Scale the distance
                distance_prob[distance] += (
                    np.abs(psi_2D[i, j]) ** 2
                )  # Assuming psi_2D is already normalized
                scaled_distances[
                    distance
                ] = scaled_distance  # Store the scaled distance
        # Normalize the distance_prob array
        distance_prob /= np.sum(distance_prob)
        # Plotting
        plt.figure(figsize=(8, 8))
        plt.plot(scaled_distances, distance_prob)  # Plot against scaled_distances
        plt.title(f"Probability Amplitude at Step {step * deltaT:.2e}")
        plt.xlabel("|x_1 - x_2| (m)")
        plt.ylabel("Probability Amplitude")
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()


# Display an animation updating time evolution of the 1D state where x axis is |x-y|
def graphic_animation_1D(
    num_frames, psi_2D_t, deltaT, frames_per_second, filename=None
):
    psi_2D_t = np.array(psi_2D_t)  # Convert to NumPy array if it's not already
    n = psi_2D_t.shape[1]  # Assuming psi_2D_t has shape (steps, n, n)
    fig, ax = plt.subplots(figsize=(8, 8))
    (line,) = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        psi_2D = psi_2D_t[frame]
        # Calculate the probability amplitude for each distance |x - y|
        distance_prob = np.zeros(n)
        for i in range(n):
            for j in range(n):
                distance = abs(i - j)
                distance_prob[distance] += (
                    np.abs(psi_2D[i, j]) ** 2
                )  # Assuming psi_2D is already normalized
        # Normalize the distance_prob array
        distance_prob /= np.sum(distance_prob)
        line.set_data(range(n), distance_prob)
        ax.set_title(f"Time: {frame * deltaT:.2e} seconds")
        return (line,)

    ax.set_xlim(0, n)
    ax.set_ylim(0, np.max(psi_2D_t))  # Set to an appropriate max value
    ax.set_xlabel("|x - y|")
    ax.set_ylabel("Probability Amplitude")
    ani = FuncAnimation(
        fig,
        update,
        frames=range(num_frames),
        repeat=False,
        interval=1000 / frames_per_second,
    )
    if filename:
        ani.save(filename, writer="ffmpeg", fps=frames_per_second)
    # Show animation
    plt.show()
