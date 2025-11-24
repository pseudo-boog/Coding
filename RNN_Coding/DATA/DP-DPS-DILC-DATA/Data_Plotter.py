from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import os


def plot_csv_comparison(path1: str, path2: str, interval: float = 7.5, show: bool = True, save_path: Optional[str] = None) -> Tuple[Figure, Axes]:
    try:
        head1 = pd.read_csv(path1, nrows=0)
        head2 = pd.read_csv(path2, nrows=0)
        
        # Load first CSV
        if {"A", "P", "B"}.issubset(head1.columns):
            df = pd.read_csv(path1, usecols=["A", "P", "B"])
        else:
            df = pd.read_csv(path1, header=None, usecols=[0, 1, 2])
            df.columns = ["A", "P", "B"]
            
        # Load second CSV
        if {"A", "P", "B"}.issubset(head2.columns):
            df2 = pd.read_csv(path2, usecols=["A", "P", "B"])
        else:
            df2 = pd.read_csv(path2, header=None, usecols=[0, 1, 2])
            df2.columns = ["A", "P", "B"]
            
    except Exception as e:
        raise ValueError(f"Error reading CSV files '{path1}' or '{path2}': {e}")

    n = len(df)
    n2 = len(df2)
    
    # Use the minimum length to avoid index errors
    min_n = min(n, n2)
    x = np.arange(min_n) * interval  # minutes, starting at 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title('Kumada Duplicate Comparison')
    ax.plot(x, df["A"][:min_n], label="PhBr", color = 'red')
    ax.plot(x, df["P"][:min_n], label="Prod", color = 'blue')
    ax.plot(x, df["B"][:min_n],  label="Grig", color = 'green')
    ax.plot(x, df2["A"][:min_n], label="PhBr2", color = 'red', linestyle='dashed')
    ax.plot(x, df2["P"][:min_n], label="Prod2", color = 'blue', linestyle='dashed')
    ax.plot(x, df2["B"][:min_n], label="Grig2", color = 'green', linestyle='dashed')

    ax.set_xlabel("Elapsed time")
    ax.set_ylabel("Normalized concentration")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(x[0] if min_n > 0 else 0, x[-1] if min_n > 0 else interval)
    ax.set_xticks(x)
    labels = [f"{int(t)}" if i % 5 == 0 else "" for i, t in enumerate(x)]
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()  # Add legend
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax

def plot_csv_plain(path1: str, interval: float = 7.5, show: bool = True, save_path: Optional[str] = None) -> Tuple[Figure, Axes]:
    try:
        head1 = pd.read_csv(path1, nrows=0)
        if {"A", "B", "P"}.issubset(head1.columns):
            df = pd.read_csv(path1, usecols=["A", "P", "B", "BP"])
        else:
            df = pd.read_csv(path1, header=None, usecols=[0, 1, 2])
            df.columns = ["A", "P", "B"]
    except Exception as e:
        raise ValueError(f"Error reading CSV '{path1}': {e}")

    n = len(df)
    x = np.arange(n) * interval  # minutes, starting at 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title('DP-DILC-Kumada')
    ax.plot(x, df["A"], label="CF3-PhBr", color='blue', linestyle='dashed', marker='o')
    ax.plot(x, df["P"], label="Prod", color='red', linestyle='dashed', marker='o')
    # ax.plot(x, (df["A"]+df["P"]), label="Mass Balance", color='purple', linestyle='dotted')
    ax.plot(x, df["B"], label="Grig", color='green', marker='o')
    # ax.plot(x, df["BP"], label="Bi-Ph", color='orange', marker='o')

    ax.set_xlabel("Elapsed Time in Minutes")
    ax.set_ylabel("Concentration (M)")
    ax.set_ylim(0, max(df[["A", "P", "B", "BP"]].max()*1.05))
    ax.set_xlim(x[0] if n > 0 else 0, x[-1] if n > 0 else interval)
    ax.set_xticks(x)
    labels = [f"{int(t)}" if i % 5 == 0 else "" for i, t in enumerate(x)]
    ax.set_xticklabels(labels, rotation=45)
    ax.legend(loc='center right')  # Add legend
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax

def rpka_plot(path: str, interval: float = 6.5, show: bool = True, save_path: Optional[str] = None) -> Tuple[Figure, Axes]:
    """Calculates the rate of decay or production across three points in a plain CSV file."""
    if interval <= 0:
        raise ValueError("Interval must be positive")
        
    try:
        # Read the CSV file with header
        head = pd.read_csv(path, nrows=0)
        
        # Try to read with named columns first
        if {"A", "P", "B"}.issubset(head.columns):
            df = pd.read_csv(path, usecols=["A", "P", "B"])
        else:
            # Fall back to reading first 3 columns (skipping Time column if present)
            df = pd.read_csv(path, header=0, usecols=[1, 2, 3])
            df.columns = ["A", "P", "B"]

        # Convert columns to numeric, coercing invalid values to NaN
        df["A"] = pd.to_numeric(df["A"], errors='coerce')
        df["P"] = pd.to_numeric(df["P"], errors='coerce')
        df["B"] = pd.to_numeric(df["B"], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna()
    except Exception as e:
        raise ValueError(f"Error reading CSV file '{path}': {e}")

    # Ensure there is enough data to calculate rates
    if len(df) < 3:
        raise ValueError("Not enough data points to calculate rates. Ensure the CSV has at least 3 valid rows.")

    n = len(df)
    x = np.arange(n) * interval  # Time points

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title('Kumada RPKA Plot')

    # Calculate rates of change
    rpka_A = []
    rpka_P = []
    rpka_B = []
    for i in range(1, n - 1):
        # Calculate rate using central difference (dy/dx = (y[i+1] - y[i-1]) / (2*dx))
        rate_A = (df["A"].iloc[i + 1] - df["A"].iloc[i - 1]) / (2 * interval)
        rate_P = (df["P"].iloc[i + 1] - df["P"].iloc[i - 1]) / (2 * interval)
        rate_B = (df["B"].iloc[i + 1] - df["B"].iloc[i - 1]) / (2 * interval)
        rpka_A.append(rate_A)
        rpka_P.append(rate_P)
        rpka_B.append(rate_B)

    # Plot the rates
    ax.plot(x[1:-1], rpka_A, label="PhBr RPKA", color='red', marker='o', linewidth=2)
    ax.plot(x[1:-1], rpka_P, label="Prod RPKA", color='blue', marker='s', linewidth=2)
    ax.plot(x[1:-1], rpka_B, label="Grig RPKA", color='green', marker='^', linewidth=2)
    
    # Set y-limits with better handling of edge cases
    if rpka_A or rpka_P or rpka_B:
        all_rates = rpka_A + rpka_P + rpka_B
        rate_min = min(all_rates)
        rate_max = max(all_rates)
        
        # Avoid flat lines when all rates are near zero
        if abs(rate_max - rate_min) < 1e-10:
            ax.set_ylim(rate_min - 0.1, rate_max + 0.1)
        else:
            margin = (rate_max - rate_min) * 0.1
            ax.set_ylim(rate_min - margin, rate_max + margin)

    ax.set_xlabel("Elapsed Time in Minutes")
    ax.set_ylabel("Rate of Change, dM/dt (M/min⁻¹)")  # Changed units to match interval
    ax.legend()
    ax.grid(True, alpha=0.3)  # Add grid for better readability
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()

    return fig, ax

if __name__ == "__main__":
    # Example usage - update paths as needed
    base_path = "/Users/dylanpyle/Coding/RNN_Code/DATA/DP-DPS-DILC-DATA"
    
    try:
        # plot_csv_comparison(
        #     path1=os.path.join(base_path, "DP-DPS-015.csv"), 
        #     path2=os.path.join(base_path, "DP-DPS-016.csv")
        # )
        
        plot_csv_plain(path1=os.path.join(base_path, "DP-DPS-021.csv"))
        
        rpka_plot(path=os.path.join(base_path, "DP-DPS-021.csv"))
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")