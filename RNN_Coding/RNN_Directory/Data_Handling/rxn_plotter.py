import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pathlib as Path

def plot_duo_mechanisms_time(mech_list, base_dir, rct_idx=0):
    """
    Plot A and P vs time for exp_0 and exp_1 for each mechanism.
    Parameters:
    mech_list: list of str -> e.g. ['M1_n','M1_pd','M1_sd','M1_cd','M1_si']
    base_dir: str -> root directory with mechanism folders
    rct_idx: int -> which rct_X to load (default 0)
    """
    # Improved figure sizing and spacing
    fig, axes = plt.subplots(1, len(mech_list), 
                            figsize=(4.5*len(mech_list), 6),  # Reduced width per subplot, increased height
                            sharey=True)
    
    if len(mech_list) == 1:
        axes = [axes]  # ensure iterable
    
    for i, (ax, mech) in enumerate(zip(axes, mech_list)):
        exp0_path = os.path.join(base_dir, mech, f"rct_{rct_idx}", "exp_0.csv")
        exp1_path = os.path.join(base_dir, mech, f"rct_{rct_idx}", "exp_1.csv")
        
        try:
            df0 = pd.read_csv(exp0_path, header=None, usecols=[0,1])
            df1 = pd.read_csv(exp1_path, header=None, usecols=[0,1])
            df0.columns, df1.columns = ["A", "P"], ["A", "P"]
        except Exception as e:
            print(f"[WARN] Skipping {mech}: {e}")
            continue
        
        # # Normalize each exp independently to [0,1]
        # df0 = df0 / df0.max()
        # df1 = df1 / df1.max()
        
        # Plot A and P vs time
        ax.plot(df0.index, df0["A"], label="exp_0 A", color="cyan", alpha=0.7, linewidth=2)
        ax.plot(df0.index, df0["P"], label="exp_0 P", color="red", alpha=0.7, linewidth=2)
        ax.plot(df1.index, df1["A"], label="exp_1 A", color="blue", alpha=0.7, linewidth=2)
        ax.plot(df1.index, df1["P"], label="exp_1 P", color="orange", alpha=0.7, linewidth=2)
        
        # Improved title and labels
        ax.set_title(mech, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Time step", fontsize=10)
        
        # Only add ylabel to the leftmost subplot
        if i == 0:
            ax.set_ylabel("Normalized concentration", fontsize=10)
        
        # Improved legend positioning
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3)
        
        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=9)
    
    # Main title with better spacing
    plt.suptitle(f"Reaction duos (rct_{rct_idx})", fontsize=16, fontweight='bold', y=0.95)
    
    # Key improvement: Better layout with more spacing
    plt.tight_layout(rect=(0.08, 0, 1, 0.92))  # Leave space for suptitle and y-label
    plt.subplots_adjust(wspace=0.15, left=0.08)  # Add horizontal spacing and left margin
    
    plt.show()

def plot_max_catI_violin(
    base_dir: str,
    mech_list: list,
    save_path: str | None = None,
    figsize: tuple = (14, 8)
):
    """Create violin plot showing distribution of max_catI_fraction values."""
    
    # Collect data (same as above)
    all_data = []
    
    for mech_name in mech_list:
        mech_dir = os.path.join(base_dir, mech_name)
        
        if not os.path.exists(mech_dir):
            continue
        
        for rct_folder in sorted(os.listdir(mech_dir)):
            if not rct_folder.startswith('rct_'):
                continue
            
            info_path = os.path.join(mech_dir, rct_folder, 'info_dict.json')
            
            if not os.path.exists(info_path):
                continue
            
            try:
                with open(info_path, 'r') as f:
                    data = json.load(f)
                
                frac_1 = data.get('max_catI_fraction_1')
                frac_2 = data.get('max_catI_fraction_2')
                
                if frac_1 is not None:
                    all_data.append({'mechanism': mech_name, 'max_catI_fraction': frac_1})
                if frac_2 is not None:
                    all_data.append({'mechanism': mech_name, 'max_catI_fraction': frac_2})
                    
            except Exception as e:
                print(f"Error reading {info_path}: {e}")
    
    if not all_data:
        print("No data found!")
        return
    
    # Organize data for violin plot
    mechanisms = sorted(set(d['mechanism'] for d in all_data))
    data_by_mech = [[d['max_catI_fraction'] for d in all_data if d['mechanism'] == mech] 
                    for mech in mechanisms]
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=figsize)
    
    parts = ax.violinplot(data_by_mech, positions=range(len(mechanisms)), 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    colors = plt.cm.tab10(np.linspace(0, 1, len(mechanisms)))  # type: ignore
    for i, pc in enumerate(parts['bodies']):  # type: ignore
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right')
    ax.set_ylabel('max_catI_fraction', fontsize=12)
    ax.set_xlabel('Mechanism', fontsize=12)
    ax.set_title('Distribution of max_catI_fraction Across Mechanisms (Violin Plot)', 
                 fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return all_data


# Usage example
# if __name__ == "__main__":
#     mech_list = [
#                 'M1_sd', 'M1_sd', 'M1_cd', 'M1_pd', 'M1_a'
#                 # 'M2_sd', 'M2_sd', 'M2_cd', 'M2_pd',
#                 # 'M3_sd', 'M3_sd', 'M3_cd', 'M3_pd'
#                 ]
    
#     base_dir = '/Users/dylanpyle/VsCode/Parrallel_0.1_catIreq_merged'
    
#     # Violin plot
#     data = plot_max_catI_violin(
#         base_dir=base_dir,
#         mech_list=mech_list,
#         save_path='catI_distribution_violin.png'
#     )

if __name__ == "__main__":
    mech_list = [
                # 'M1_n', 'M1_cd', 'M1_scd', 'M1_pcd', 
                # 'M1_c1d', 'M1_pc1d', 'M1_sc1d', 'M2_n', 
                # 'M2_cd', 'M2_scd', 'M2_pcd', 'M2_c1d', 
                # 'M2_pc1d', 'M2_sc1d', 'M2_c2d', 'M2_pc2d', 
                # 'M2_sc2d', 'M3_n', 'M3_cd', 'M3_scd', 
                # 'M3_pcd', 'M3_c1d', 'M3_pc1d', 'M3_sc1d', 
                # 'M3_c2d', 'M3_pc2d', 'M3_sc2d'
                ]
    base_dir = "/Users/dylanpyle/Coding/RNN_Code/DATA/M1M2M3_data_Normalized"
    plot_duo_mechanisms_time(mech_list, base_dir, rct_idx=3)

# Read the duoconcat CSV file (no headers, 4 columns)
# if __name__ == "__main__":
#     data = pd.read_csv('M1_pd_duoconcat.csv', header=None, names=['SM_Exp0', 'P_Exp0', 'SM_Exp1', 'P_Exp1'])

#     # Create time index
#     time_points = np.arange(len(data))

#     # Create the plot
#     plt.figure(figsize=(12, 8))

#     # Plot all 4 lines
#     plt.plot(time_points, data['SM_Exp0'], color = 'cyan', linewidth=2, label='SM Exp0', marker='o')
#     plt.plot(time_points, data['P_Exp0'], color = 'red', linewidth=2, label='P Exp0', marker='s')
#     plt.plot(time_points, data['SM_Exp1'], color = 'blue', linewidth=2, label='SM Exp1')
#     plt.plot(time_points, data['P_Exp1'], color = 'orange', linewidth=2, label='P Exp1')

#     # Customize the plot
#     plt.xlabel('Time Point Index', fontsize=12)
#     plt.ylabel('Normalized Concentration', fontsize=12)
#     plt.title('[Os] Data - DuoConcat Data', fontsize=14, fontweight='bold')
#     plt.legend(fontsize=10)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     # Show statistics
#     print("=== DATA STATS ===")
#     print(f"Shape: {data.shape}")
#     print(f"SM_Exp0: {data['SM_Exp0'].min():.3f} to {data['SM_Exp0'].max():.3f}")
#     print(f"P_Exp0: {data['P_Exp0'].min():.3f} to {data['P_Exp0'].max():.3f}")
#     print(f"SM_Exp1: {data['SM_Exp1'].min():.3f} to {data['SM_Exp1'].max():.3f}")
#     print(f"P_Exp1: {data['P_Exp1'].min():.3f} to {data['P_Exp1'].max():.3f}")

#     # Save and show
#     plt.savefig('M1_pd_plot.png', dpi=300, bbox_inches='tight')
#     plt.show()


# if __name__ == "__main__":
#     data = pd.read_csv('/Users/dylanpyle/VsCode/Dylan Kumada DILC - DP-DPS-VTNA-002+006.csv', header=0)
#     time_points = np.arange(len(data))

#     """Need to Normalize the data!!"""

#     plt.figure(figsize=(12, 8))

#     plt.plot(time_points, data.iloc[:, 0], color='blue', linewidth=2, label='SM 0.5% cat loading')
#     plt.plot(time_points, data.iloc[:, 1], color='red', linewidth=2, label='Prod 0.5% cat loading')
#     plt.plot(time_points, data.iloc[:, 2], linestyle='dashed', color='blue', linewidth=2, label='SM 1% cat loading')
#     plt.plot(time_points, data.iloc[:, 3], linestyle='dashed', color='red', linewidth=2, label='Prod 1% cat loading')


#     plt.xlabel('Time Point (sample every 6 minutes)', fontsize=12)
#     plt.ylabel('Concentrations', fontsize=12)
#     plt.title('Pd Cross Couple DILC', fontsize=20, fontweight='bold')
#     plt.legend(fontsize=15, loc='center right')
#     # plt.tight_layout()

#     # Save and show
#     plt.savefig('M1_pd_plot.png', dpi=300, bbox_inches='tight')
#     plt.show()

