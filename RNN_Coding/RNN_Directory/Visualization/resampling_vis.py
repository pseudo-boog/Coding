import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_csv_with_resampling(csv_file):
    """
    Reads a CSV file and plots both the full dataset and resampled versions.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file where rows are time points and columns are 
        chemical concentrations (column 0 = A, column 1 = P)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, header=None, names=['A', 'P'])
    
    # Get the full data
    A_full = df['A'].values
    P_full = df['P'].values
    
    # Create time points for the full dataset
    time_full = np.arange(len(A_full))

    # Resample every 4th point for Chemical A and P
    resample_indices_4 = np.arange(0, len(A_full), 4, dtype=int)
    A_resampled_4 = A_full[resample_indices_4]
    P_resampled_4 = P_full[resample_indices_4]
    time_resampled_4 = resample_indices_4
    # Scale resampled time to be 1/4 the length
    time_resampled_scaled_4 = time_resampled_4 * 0.25
    
    # Resample every 10th point (uniformly spaced to get 10 points)
    resample_indices_10 = np.linspace(0, len(A_full) - 1, 10, dtype=int)
    A_resampled_10 = A_full[resample_indices_10]
    P_resampled_10 = P_full[resample_indices_10]
    time_resampled_10 = resample_indices_10
    # Scale resampled time to be 1/10 the length
    time_resampled_scaled_10 = time_resampled_10 * 0.1
    
    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Chemical A over time
    ax1.plot(time_full, A_full, 'b-', linewidth=2, label='Full data (100 points)', alpha=0.7)
    ax1.plot(time_full, A_full, 'bo', markersize=3, alpha=0.5)

    ax1.plot(time_resampled_scaled_4, A_resampled_4, 'r-', linewidth=2,
            label='Resampled (25 points, every 4th, 1/4 length)', alpha=0.7)
    ax1.plot(time_resampled_scaled_4, A_resampled_4, 'ro', markersize=5)

    ax1.plot(time_resampled_scaled_10, A_resampled_10, 'g-', linewidth=2,
            label='Resampled (10 points, 1/10 length)', alpha=0.7)
    ax1.plot(time_resampled_scaled_10, A_resampled_10, 'go', markersize=6)

    ax1.axvline(x=len(time_full) * 0.25, color='red', linestyle='--',
               alpha=0.3, linewidth=1, label='1/4 length boundary')
    ax1.axvline(x=len(time_full) * 0.1, color='green', linestyle='--',
               alpha=0.3, linewidth=1, label='1/10 length boundary')
    
    ax1.set_xlabel('Time Point', fontsize=12)
    ax1.set_ylabel('A Concentration', fontsize=12)
    ax1.set_title('Sm (A): Full vs Resampled Versions', fontsize=14)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Chemical P over time
    ax2.plot(time_full, P_full, 'b-', linewidth=2, label='Full data (100 points)', alpha=0.7)
    ax2.plot(time_full, P_full, 'bo', markersize=3, alpha=0.5)
    
    ax2.plot(time_resampled_scaled_4, P_resampled_4, 'r-', linewidth=2, 
            label='Resampled (25 points, every 4th, 1/4 length)', alpha=0.7)
    ax2.plot(time_resampled_scaled_4, P_resampled_4, 'ro', markersize=5)
    
    ax2.plot(time_resampled_scaled_10, P_resampled_10, 'g-', linewidth=2,
            label='Resampled (10 points, 1/10 length)', alpha=0.7)
    ax2.plot(time_resampled_scaled_10, P_resampled_10, 'go', markersize=6)
    
    ax2.axvline(x=len(time_full) * 0.25, color='red', linestyle='--',
               alpha=0.3, linewidth=1, label='1/4 length boundary')
    ax2.axvline(x=len(time_full) * 0.1, color='green', linestyle='--',
               alpha=0.3, linewidth=1, label='1/10 length boundary')
    
    ax2.set_xlabel('Time Point', fontsize=12)
    ax2.set_ylabel('P Concentration', fontsize=12)
    ax2.set_title('Product: Full vs Resampled Versions', fontsize=14)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"Full dataset: {len(time_full)} time points")
    print(f"Resampled dataset (every 4th): {len(time_resampled_4)} time points")
    print(f"Resampled dataset (10 points): {len(time_resampled_10)} time points")
    print(f"Time range (full): [0, {len(time_full)-1}]")
    print(f"Time range (every 4th, scaled): [0, {time_resampled_scaled_4.max():.2f}]")
    print(f"Time range (10 points, scaled): [0, {time_resampled_scaled_10.max():.2f}]")
    print(f"\nChemical A range: [{A_full.min():.4f}, {A_full.max():.4f}]")
    print(f"Chemical P range: [{P_full.min():.4f}, {P_full.max():.4f}]")

# Example usage:
if __name__ == "__main__":
    plot_csv_with_resampling('/Users/dylanpyle/VsCode/RNN_Code/DATA/PT_activation/M1_n/rct_4949/exp_0.csv')