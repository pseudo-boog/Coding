import matplotlib.pyplot as plt
import numpy as np
from RNN_Module import get_data_filtered_duos, mech_list, resample_sequence, ResamplingStrategy

def visualize_mechanism_uniform_resampling(
    mech_list,
    start_iteration=0,
    end_iteration=100,
    target_length=20,
    num_reactions=4,
    data_base_dir="/Users/dylanpyle/VsCode/RNN_Code/DATA/training_0.1_catIreq",
    json_base_dir="/Users/dylanpyle/VsCode/test_0.1_catIreq",
    figsize=(15, 10)
):
    """
    Visualize uniform resampling on multiple reactions from mechanism list.
    
    Parameters:
    -----------
    mech_list : list
        List of mechanism names (e.g., ['M1_n', 'M1_cd', ...])
    start_iteration : int
        Starting iteration number for data loading
    end_iteration : int
        Ending iteration number for data loading
    target_length : int
        Target length for uniform resampling
    num_reactions : int
        Number of reactions to visualize
    data_base_dir : str
        Base directory for reaction data
    json_base_dir : str
        Base directory for JSON metadata
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Load data using your existing function
    print(f"Loading data from iterations {start_iteration} to {end_iteration}...")
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        min_time_points=8,
        data_base_dir=data_base_dir,
        json_base_dir=json_base_dir,
        verbose=False
    )
    
    if len(X_list) == 0:
        raise ValueError("No valid data found in the specified range.")
    
    print(f"Loaded {len(X_list)} sequences")
    
    # Limit to num_reactions
    num_to_plot = min(num_reactions, len(X_list))
    
    # Determine subplot grid
    n_rows = int(np.ceil(num_to_plot / 2))
    n_cols = 2 if num_to_plot > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if num_to_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each reaction
    for idx in range(num_to_plot):
        ax = axes[idx]
        
        # Get the sequence data
        data = X_list[idx]
        mechanism_name = y_name_list[idx]
        
        original_length = len(data)
        
        # Perform uniform resampling
        indices = np.linspace(0, original_length - 1, target_length, dtype=int)
        resampled_data = data[indices]
        
        original_indices = np.arange(original_length)
        
        # Plot - using first feature (A1) from the duo dataset
        # data shape is (T, 4) where columns are [A1, P1, A2, P2]
        ax.plot(original_indices, data[:, 0], 'b-', alpha=0.3, 
                label='Original (A1)', linewidth=1)
        ax.plot(indices, resampled_data[:, 0], 'ro-', 
                label='Resampled', markersize=6, linewidth=2)
        
        ax.set_title(f'{mechanism_name} - Reaction {idx+1}\n'
                     f'Uniform Resampling ({original_length} → {target_length})', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Absorbance (A1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_to_plot, len(axes)):
        axes[idx].axis('off')
    plt.show()
    
    return fig


def visualize_specific_mechanisms(
    mechanisms_to_plot,
    start_iteration=0,
    end_iteration=100,
    target_length=20,
    data_base_dir="/Users/dylanpyle/VsCode/RNN_Code/DATA/training_0.1_catIreq",
    json_base_dir="/Users/dylanpyle/VsCode/test_0.1_catIreq",
    figsize=(15, 10)
):
    """
    Visualize uniform resampling for specific mechanisms.
    
    Parameters:
    -----------
    mechanisms_to_plot : list
        List of specific mechanism names to plot (e.g., ['M1_n', 'M2_a'])
    start_iteration : int
        Starting iteration number
    end_iteration : int
        Ending iteration number
    target_length : int
        Target length for resampling
    data_base_dir : str
        Base directory for data
    json_base_dir : str
        Base directory for JSON
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Load data for specific mechanisms
    print(f"Loading data for mechanisms: {mechanisms_to_plot}")
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mechanisms_to_plot,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        min_time_points=8,
        data_base_dir=data_base_dir,
        json_base_dir=json_base_dir,
        verbose=False
    )
    
    if len(X_list) == 0:
        raise ValueError("No valid data found for specified mechanisms.")
    
    print(f"Loaded {len(X_list)} sequences")
    
    # Group by mechanism
    mechanism_sequences = {mech: [] for mech in mechanisms_to_plot}
    for seq, name in zip(X_list, y_name_list):
        if name in mechanism_sequences:
            mechanism_sequences[name].append(seq)
    
    # Plot one example from each mechanism
    num_to_plot = len(mechanisms_to_plot)
    n_rows = int(np.ceil(num_to_plot / 2))
    n_cols = 2 if num_to_plot > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if num_to_plot == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, mechanism_name in enumerate(mechanisms_to_plot):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        sequences = mechanism_sequences[mechanism_name]
        if len(sequences) == 0:
            ax.text(0.5, 0.5, f'No data for {mechanism_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(mechanism_name)
            continue
        
        # Use first sequence for this mechanism
        data = sequences[0]
        original_length = len(data)
        
        # Perform uniform resampling
        indices = np.linspace(0, original_length - 1, target_length, dtype=int)
        resampled_data = data[indices]
        
        original_indices = np.arange(original_length)
        
        # Plot
        ax.plot(original_indices, data[:, 0], 'b-', alpha=0.3, 
                label='Original (A1)', linewidth=1)
        ax.plot(indices, resampled_data[:, 0], 'ro-', 
                label='Resampled', markersize=6, linewidth=2)
        
        ax.set_title(f'{mechanism_name}\n'
                     f'Uniform Resampling ({original_length} → {target_length})', 
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Absorbance (A1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_to_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compare_features_uniform_resampling(
    mech_list,
    start_iteration=0,
    end_iteration=100,
    target_length=20,
    reaction_idx=0,
    data_base_dir="/Users/dylanpyle/VsCode/RNN_Code/DATA/training_0.1_catIreq",
    json_base_dir="/Users/dylanpyle/VsCode/test_0.1_catIreq",
    figsize=(15, 8)
):
    """
    Visualize uniform resampling across all 4 features (A1, P1, A2, P2) for one reaction.
    
    Parameters:
    -----------
    mech_list : list
        List of mechanism names
    start_iteration : int
        Starting iteration
    end_iteration : int
        Ending iteration
    target_length : int
        Target length for resampling
    reaction_idx : int
        Index of reaction to visualize
    data_base_dir : str
        Base directory for data
    json_base_dir : str
        Base directory for JSON
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Load data
    print("Loading data...")
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        min_time_points=8,
        data_base_dir=data_base_dir,
        json_base_dir=json_base_dir,
        verbose=False
    )
    
    if reaction_idx >= len(X_list):
        raise ValueError(f"Reaction index {reaction_idx} out of range (only {len(X_list)} sequences loaded)")
    
    data = X_list[reaction_idx]
    mechanism_name = y_name_list[reaction_idx]
    original_length = len(data)
    
    # Perform uniform resampling
    indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    resampled_data = data[indices]
    
    original_indices = np.arange(original_length)
    
    # Plot all 4 features
    feature_names = ['A1 (Exp 0)', 'P1 (Exp 0)', 'A2 (Exp 1)', 'P2 (Exp 1)']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for feat_idx in range(4):
        ax = axes[feat_idx]
        
        ax.plot(original_indices, data[:, feat_idx], 'b-', alpha=0.3, 
                label='Original', linewidth=1)
        ax.plot(indices, resampled_data[:, feat_idx], 'ro-', 
                label='Resampled', markersize=6, linewidth=2)
        
        ax.set_title(f'{feature_names[feat_idx]}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{mechanism_name} - All Features\n'
                 f'Uniform Resampling ({original_length} → {target_length})',
                 fontsize=13, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    plt.show()
    
    return fig



mech_list = ['M1_n']

fig = visualize_mechanism_uniform_resampling(
    mech_list=mech_list,
    start_iteration=0,
    end_iteration=2,
    target_length=15,
    num_reactions=1
)

