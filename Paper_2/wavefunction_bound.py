#######
# Import relevant libraries
#######

import marshal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import structlog
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
from scipy.sparse.linalg import eigs, eigsh
import matplotlib.lines as mlines  

def main():
    L = float(input("Enter the value for L: "))
    d2 = float(input("Enter the value for d2: "))
    d3 = float(input("Enter the value for d3: "))

    x1 = np.linspace(0.1, L - 0.1, 1000)
    x2 = np.linspace(0.1, L - 0.1, 1000)

    X1, X2 = np.meshgrid(x1, x2)

    Z1 = (
        1 / (4 * X1**2) +
        1 / (4 * X2**2) +
        1 / (4 * (L - X1) ** 2) +
        1 / (4 * (L - X2) ** 2)
    )
    Z2 = 1 / ((L - d2) ** 2) + 1 / ((L + d2) ** 2) + 2 / L**2
    Z3 = 1 / ((L - d3) ** 2) + 1 / ((L + d3) ** 2) + 2 / L**2

    plt.figure(figsize=(10, 10))  # Adjusted figure size
    plt.scatter(X1[Z1 < Z2], X2[Z1 < Z2], c="lightblue", s=10, label="Reduced QZE confinement region ($d_{i+1} = 6$)")
    plt.scatter(X1[(Z1 > Z2) & (Z1 < Z3)], X2[(Z1 > Z2) & (Z1 < Z3)], c="lightgreen", s=10, label="Original QZE confinement region ($d_i = 7$)")
    # plt.plot(x1, x1, 'k--', markersize=10, label='Maximum Coulomb repulsive force line ($x_1=x_2$)')

    current_font_size = 12
    plt.xlabel("$x_1$ - position of ion 1", fontsize=current_font_size + 5)
    plt.ylabel("$x_2$ - position of ion 2", fontsize=current_font_size + 5)
    plt.title("Wavefunction Boundary change due to Reduction in QZE Confinement Region", pad=20, fontsize=current_font_size + 5)

    # Manually adjust subplot parameters to reduce white space
    plt.subplots_adjust(bottom=0.25, top=0.95)  # Adjust these values as needed

    # Adjust legend positioning
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1, fontsize=current_font_size + 5)

    plt.xlim(0, L)
    plt.ylim(0, L)
    plt.grid(True)
    plt.xticks(fontsize=current_font_size)
    plt.yticks(fontsize=current_font_size)

    # No longer using tight_layout here to manually control spacing
    plt.show()

if __name__ == "__main__":
    main()