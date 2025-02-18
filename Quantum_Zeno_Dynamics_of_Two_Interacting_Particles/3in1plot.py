#########################
#This code read the data of all graphs (leakage vs. time) and plot them together in logarithmic scale. 
#The associated data is generated in the code "Leakage_Time.py". 
######################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

# File names and corresponding labels
files = ["leakage_vs_time1.csv", "leakage_vs_time2.csv", "leakage_vs_time3.csv"]
labels = [r"$d=10^{-12}$ m", r"$d=10^{-11}$ m", r"$d=10^{-10}$ m"]

# Function to format x-axis ticks
def log_formatter(x, pos):
    return f"{int(np.log10(x))}" if x != 0 else "0"

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop through each file and its corresponding label
for file, label in zip(files, labels):
    data = pd.read_csv(file)
    time = data.iloc[:, 0]  # First column: time
    leakage = data.iloc[:, 1]  # Second column: leakage probability
    
    # Append (0, 0) to ensure the curve touches the origin
    time = np.append(0, time)  # Add 0 to the start of time
    leakage = np.append(0, leakage)  # Add 0 to the start of leakage probability
    
    # Plot the data
    plt.plot(time, leakage, label=label)

# Set logarithmic scale for time axis
plt.xscale("symlog", linthresh=1e-20)  # Symmetric log scale with a small threshold

# Customize the x-axis ticks
plt.gca().xaxis.set_major_formatter(FuncFormatter(log_formatter))

# Add labels, title, and legend
plt.xlabel("Log[Time (s)]")
plt.ylabel("Leakage Probability")
plt.title("Leakage Probability vs Time")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc="upper left")  # Position legend at top-left corner

# Show the plot
plt.tight_layout()
plt.show()
