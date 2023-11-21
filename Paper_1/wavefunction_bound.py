# Please save files here: /home/varqa/schrodinger/main_v1.py (\\wsl.localhost\Ubuntu\home\varqa\schrodinger)

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

import quantum_library as ql

plt.style.use(["science", "notebook"])


def main():
    L = float(input("Enter the value for L: "))
    d = float(input("Enter the value for d: "))

    x1 = np.linspace(0.1, L - 0.1, 1000)
    x2 = np.linspace(0.1, L - 0.1, 1000)

    X1, X2 = np.meshgrid(x1, x2)

    # Calculating the given expressions
    Z1 = (
        1 / (4 * X1**2)
        + 1 / (4 * X2**2)
        + 1 / (4 * (L - X1) ** 2)
        + 1 / (4 * (L - X2) ** 2)
    )
    Z2 = 1 / ((L - d) ** 2) + 1 / ((L + d) ** 2) + 2 / L**2

    condition_met = Z1 < Z2

    # Find the furthest point
    distances = np.sqrt((X1 - L / 2) ** 2 + (X2 - L / 2) ** 2)
    max_distance = np.max(distances[condition_met])
    print(f"The furthest point from (L/2, L/2) is {max_distance:.3f} units away.")

    plt.figure(figsize=(10, 10))
    plt.scatter(X1[condition_met], X2[condition_met], c="gray", s=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Shaded regions have F_Total < F_Total^Max")

    # Plotting center point
    plt.plot(L / 2, L / 2, "ro", label="Center (L/2, L/2)")
    # Set x and y axis limits
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
