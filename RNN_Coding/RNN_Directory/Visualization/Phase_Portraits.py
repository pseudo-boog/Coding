import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.widgets import Slider
from typing import Dict, List, Tuple, Any


def parse_mechanism(reaction_strings: List[str]) -> Dict[str, Any]:
    """
    Parse a list of reaction strings and return reaction information.
    
    Args:
        reaction_strings: List of reaction strings, e.g., ["A + cat -> cat1", "cat1 -> cat + P"]
    
    Returns:
        Dictionary containing:
        - 'reactions': List of all reaction tuples (reactants, products)
        - 'species': Set of all unique species
        - 'num_reactions': Total number of reactions
        - 'k_dict': Dictionary with k1, k2, ... for each reaction
    """
    reactions = []
    species_set = set()
    
    # Parse all reactions
    for i, rxn_str in enumerate(reaction_strings):
        # Remove extra whitespace
        rxn_str = rxn_str.strip()
        
        # Split by ->
        if '->' not in rxn_str:
            raise ValueError(f"Invalid reaction format: '{rxn_str}'. Must contain '->'")
        
        parts = rxn_str.split('->')
        if len(parts) != 2:
            raise ValueError(f"Invalid reaction format: '{rxn_str}'. Must have exactly one '->'")
        
        reactants_str = parts[0].strip()
        products_str = parts[1].strip()
        
        # Parse reactants and products
        reactants = [s.strip() for s in reactants_str.split('+')]
        products = [s.strip() for s in products_str.split('+')]
        
        # Add to species set
        species_set.update(reactants)
        species_set.update(products)
        
        reactions.append({
            'reactants': reactants,
            'products': products,
            'index': i
        })
    
    # Create k_dict with one rate constant per reaction
    k_dict = {}
    for i in range(len(reactions)):
        k_dict[f'k{i+1}'] = 0.0  # Rate constant (will be set by user/sliders)
    
    return {
        'reactions': reactions,
        'species': species_set,
        'num_reactions': len(reactions),
        'k_dict': k_dict
    }


def print_mechanism_info(mechanism_info: Dict[str, Any]) -> None:
    """
    Print formatted information about the parsed mechanism.
    
    Args:
        mechanism_info: Dictionary returned from parse_mechanism()
    """
    print("\n" + "="*70)
    print("MECHANISM INFORMATION")
    print("="*70)
    
    print(f"\nSpecies detected: {sorted(mechanism_info['species'])}")
    print(f"Total reactions: {mechanism_info['num_reactions']}")
    
    print("\n--- Reactions ---")
    for rxn in mechanism_info['reactions']:
        reactants = ' + '.join(rxn['reactants'])
        products = ' + '.join(rxn['products'])
        k = f"k{rxn['index']+1}"
        print(f"  R{rxn['index']+1}: {reactants} → {products}  ({k})")
    
    print("\n" + "="*70 + "\n")


# Define the catalytic reaction system with REVERSIBLE reactions
# Core mechanism with 3 forward and 3 reverse reactions, plus 2 side reactions
# R1: A + cat → cat1         (k1_f)
# R2: cat1 → A + cat         (k2_f) [reverse of R1]
# R3: cat1 → cat + P         (k3_f)
# R4: P + cat → cat1         (k4_f) [reverse of R3]
# R5: cat + P → catI         (k5_f)
# R6: catI → cat + P         (k6_f) [reverse of R5]
# R7: A + cat → catI         (k7_f) [direct inhibition]
# R8: catI → A + cat         (k8_f) [reverse of R7]

def reaction_system(state, t, k_values):
    """
    Define the system of ODEs for the catalytic reaction with 8 reactions
    
    state: [A, P, cat1, catI]
    k_values: Dictionary with k1, k2, ..., k8 rate constants
    
    Reactions:
    R1: A + cat → cat1       (k1)
    R2: cat1 → A + cat       (k2)
    R3: cat1 → cat + P       (k3)
    R4: P + cat → cat1       (k4)
    R5: cat + P → catI       (k5)
    R6: catI → cat + P       (k6)
    R7: A + cat → catI       (k7)
    R8: catI → A + cat       (k8)
    """
    A, P, cat1, catI = state
    cat_total = k_values.get('cat_total', 1.0)
    
    # Conservation: cat = cat_total - cat1 - catI
    cat = cat_total - cat1 - catI
    
    # Prevent negative concentrations
    cat = max(0, cat)
    
    # Extract rate constants
    k1 = k_values.get('k1', 0.0)
    k2 = k_values.get('k2', 0.0)
    k3 = k_values.get('k3', 0.0)
    k4 = k_values.get('k4', 0.0)
    k5 = k_values.get('k5', 0.0)
    k6 = k_values.get('k6', 0.0)
    k7 = k_values.get('k7', 0.0)
    k8 = k_values.get('k8', 0.0)
    
    # Reaction rates
    # R1: A + cat → cat1
    r1 = k1 * A * cat
    
    # R2: cat1 → A + cat
    r2 = k2 * cat1
    
    # R3: cat1 → cat + P
    r3 = k3 * cat1
    
    # R4: P + cat → cat1
    r4 = k4 * P * cat
    
    # R5: cat + P → catI
    r5 = k5 * cat * P
    
    # R6: catI → cat + P
    r6 = k6 * catI
    
    # R7: A + cat → catI
    r7 = k7 * A * cat
    
    # R8: catI → A + cat
    r8 = k8 * catI
    
    # ODEs
    dA_dt = -r1 + r2 - r7 + r8
    dP_dt = r3 - r4 - r5 + r6
    dcat1_dt = r1 - r2 - r3 + r4
    dcatI_dt = r5 - r6 + r7 - r8
    
    return [dA_dt, dP_dt, dcat1_dt, dcatI_dt]


def compute_derivatives(A, P, cat1, catI, k_values):
    """
    Compute derivatives at a given point for vector field calculation (8 reactions)
    """
    cat_total = k_values.get('cat_total', 1.0)
    cat = cat_total - cat1 - catI
    cat = max(0, cat)
    
    # Extract rate constants
    k1 = k_values.get('k1', 0.0)
    k2 = k_values.get('k2', 0.0)
    k3 = k_values.get('k3', 0.0)
    k4 = k_values.get('k4', 0.0)
    k5 = k_values.get('k5', 0.0)
    k6 = k_values.get('k6', 0.0)
    k7 = k_values.get('k7', 0.0)
    k8 = k_values.get('k8', 0.0)
    
    # Reaction rates
    r1 = k1 * A * cat
    r2 = k2 * cat1
    r3 = k3 * cat1
    r4 = k4 * P * cat
    r5 = k5 * cat * P
    r6 = k6 * catI
    r7 = k7 * A * cat
    r8 = k8 * catI
    
    # Derivatives
    dA_dt = -r1 + r2 - r7 + r8
    dP_dt = r3 - r4 - r5 + r6
    dcat1_dt = r1 - r2 - r3 + r4
    dcatI_dt = r5 - r6 + r7 - r8
    
    return dA_dt, dP_dt, dcat1_dt, dcatI_dt


def interactive_2d_portraits(mechanism: List[str] | None = None):
    """
    Create interactive 2D phase portraits with sliders for all parameters
    
    Args:
        mechanism: List of reaction strings (e.g., ["A + cat -> cat1", "cat1 -> cat + P"])
                   If provided, prints mechanism info and creates sliders for all reactions.
                   If None, uses default 8-reaction mechanism.
    """
    # Parse mechanism and create dynamic k_values
    if mechanism is not None:
        mech_info = parse_mechanism(mechanism)
        print_mechanism_info(mech_info)
        num_reactions = mech_info['num_reactions']
    else:
        # Default mechanism with 8 reactions
        mechanism = [
            "A + cat -> cat1",
            "cat1 -> A + cat",
            "cat1 -> cat + P",
            "P + cat -> cat1",
            "cat + P -> catI",
            "catI -> cat + P",
            "A + cat -> catI",
            "catI -> A + cat"
        ]
        mech_info = parse_mechanism(mechanism)
        num_reactions = 8
    
    # Initial parameters for k-values
    k_init_dict = {}
    for i in range(1, num_reactions + 1):
        if i <= 3:
            k_init_dict[f'k{i}'] = 0.5  # Forward reactions
        elif i <= 6:
            k_init_dict[f'k{i}'] = 0.1  # Reverse reactions (slower)
        else:
            k_init_dict[f'k{i}'] = 0.0  # Side reactions (off by default)
    
    cat_total_init = 1.0
    A0_init = 2.0
    t_max = 30
    
    # Create figure with 6 subplots (3x2 grid)
    # Calculate figure height based on number of sliders
    num_slider_rows = num_reactions + 2  # all reactions + cat_total + A0
    base_plot_height = 12
    slider_height_needed = num_slider_rows * 0.35  # Approximate height per slider row
    fig_height = base_plot_height + slider_height_needed
    
    fig = plt.figure(figsize=(16, fig_height))
    # Adjust spacing to accommodate all sliders without overlap
    bottom_margin = 0.04 + (num_slider_rows * 0.04)  # Reserve space for sliders
    plt.subplots_adjust(left=0.08, bottom=bottom_margin, right=0.96, top=0.94, hspace=0.40, wspace=0.30)
    
    # Create subplots in 3x2 grid
    ax1 = fig.add_subplot(3, 2, 1)  # A vs cat1
    ax2 = fig.add_subplot(3, 2, 2)  # A vs P
    ax3 = fig.add_subplot(3, 2, 3)  # P vs cat1
    ax4 = fig.add_subplot(3, 2, 4)  # A vs catI
    ax5 = fig.add_subplot(3, 2, 5)  # P vs catI
    ax6 = fig.add_subplot(3, 2, 6)  # cat1 vs catI
    
    t = np.linspace(0, t_max, 1500)
    
    def update_plot(k_dict):
        # Initial condition - cat1, catI, and P always start at zero
        initial_state = [k_dict.get('A0', A0_init), 0.0, 0.0, 0.0]
        
        # Solve trajectory with all 8 reactions
        solution = odeint(reaction_system, initial_state, t, args=(k_dict,))
        A_sol, P_sol, cat1_sol, catI_sol = solution.T
        
        cat_total = k_dict.get('cat_total', cat_total_init)
        
        # Define ranges for vector fields
        A_range = np.linspace(0.01, 2.0, 12)
        P_range = np.linspace(0, 2.0, 12)
        cat1_range = np.linspace(0.001, cat_total*0.99, 12)
        catI_range = np.linspace(0.001, cat_total*0.99, 12)
        
        # ========== Plot 1: A vs cat1 ==========
        ax1.clear()
        A_mesh, cat1_mesh = np.meshgrid(A_range, cat1_range)
        dA = np.zeros_like(A_mesh)
        dcat1 = np.zeros_like(cat1_mesh)
        
        for i in range(A_mesh.shape[0]):
            for j in range(A_mesh.shape[1]):
                dA_dt, _, dcat1_dt, _ = compute_derivatives(
                    A_mesh[i, j], 0, cat1_mesh[i, j], 0, k_dict)
                dA[i, j] = dA_dt
                dcat1[i, j] = dcat1_dt
        
        magnitude = np.sqrt(dA**2 + dcat1**2)
        magnitude[magnitude == 0] = 1
        
        ax1.quiver(A_mesh, cat1_mesh, dA/magnitude, dcat1/magnitude, 
                  magnitude, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax1.plot(A_sol, cat1_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax1.scatter(A_sol[0], cat1_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax1.scatter(A_sol[-1], cat1_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax1.set_xlabel('[A]', fontsize=10, fontweight='bold')
        ax1.set_ylabel('[cat1]', fontsize=10, fontweight='bold')
        ax1.set_title('A vs cat1', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 2.0)
        ax1.set_ylim(0, cat_total)
        
        # ========== Plot 2: A vs P ==========
        ax2.clear()
        A_mesh2, P_mesh = np.meshgrid(A_range, P_range)
        dA2 = np.zeros_like(A_mesh2)
        dP = np.zeros_like(P_mesh)
        
        for i in range(A_mesh2.shape[0]):
            for j in range(A_mesh2.shape[1]):
                dA_dt, dP_dt, _, _ = compute_derivatives(
                    A_mesh2[i, j], P_mesh[i, j], cat_total*0.3, 0, k_dict)
                dA2[i, j] = dA_dt
                dP[i, j] = dP_dt
        
        magnitude2 = np.sqrt(dA2**2 + dP**2)
        magnitude2[magnitude2 == 0] = 1
        
        ax2.quiver(A_mesh2, P_mesh, dA2/magnitude2, dP/magnitude2, 
                  magnitude2, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax2.plot(A_sol, P_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax2.scatter(A_sol[0], P_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax2.scatter(A_sol[-1], P_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax2.set_xlabel('[A]', fontsize=10, fontweight='bold')
        ax2.set_ylabel('[P]', fontsize=10, fontweight='bold')
        ax2.set_title('A vs P', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 2.0)
        ax2.set_ylim(0, 2.0)
        
        # ========== Plot 3: P vs cat1 ==========
        ax3.clear()
        P_mesh2, cat1_mesh2 = np.meshgrid(P_range, cat1_range)
        dP2 = np.zeros_like(P_mesh2)
        dcat1_2 = np.zeros_like(cat1_mesh2)
        
        A0 = k_dict.get('A0', A0_init)
        for i in range(P_mesh2.shape[0]):
            for j in range(P_mesh2.shape[1]):
                _, dP_dt, dcat1_dt, _ = compute_derivatives(
                    A0*0.5, P_mesh2[i, j], cat1_mesh2[i, j], 0, k_dict)
                dP2[i, j] = dP_dt
                dcat1_2[i, j] = dcat1_dt
        
        magnitude3 = np.sqrt(dP2**2 + dcat1_2**2)
        magnitude3[magnitude3 == 0] = 1
        
        ax3.quiver(P_mesh2, cat1_mesh2, dP2/magnitude3, dcat1_2/magnitude3, 
                  magnitude3, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax3.plot(P_sol, cat1_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax3.scatter(P_sol[0], cat1_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax3.scatter(P_sol[-1], cat1_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax3.set_xlabel('[P]', fontsize=10, fontweight='bold')
        ax3.set_ylabel('[cat1]', fontsize=10, fontweight='bold')
        ax3.set_title('P vs cat1', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, cat_total)
        
        # ========== Plot 4: A vs catI ==========
        ax4.clear()
        A_mesh3, catI_mesh = np.meshgrid(A_range, catI_range)
        dA3 = np.zeros_like(A_mesh3)
        dcatI = np.zeros_like(catI_mesh)
        
        for i in range(A_mesh3.shape[0]):
            for j in range(A_mesh3.shape[1]):
                dA_dt, _, _, dcatI_dt = compute_derivatives(
                    A_mesh3[i, j], 1.0, 0, catI_mesh[i, j], k_dict)
                dA3[i, j] = dA_dt
                dcatI[i, j] = dcatI_dt
        
        magnitude4 = np.sqrt(dA3**2 + dcatI**2)
        magnitude4[magnitude4 == 0] = 1
        
        ax4.quiver(A_mesh3, catI_mesh, dA3/magnitude4, dcatI/magnitude4, 
                  magnitude4, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax4.plot(A_sol, catI_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax4.scatter(A_sol[0], catI_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax4.scatter(A_sol[-1], catI_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax4.set_xlabel('[A]', fontsize=10, fontweight='bold')
        ax4.set_ylabel('[catI] (Inhibited)', fontsize=10, fontweight='bold')
        ax4.set_title('A vs catI', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 2.0)
        ax4.set_ylim(0, cat_total)
        
        # ========== Plot 5: P vs catI ==========
        ax5.clear()
        P_mesh3, catI_mesh2 = np.meshgrid(P_range, catI_range)
        dP3 = np.zeros_like(P_mesh3)
        dcatI2 = np.zeros_like(catI_mesh2)
        
        for i in range(P_mesh3.shape[0]):
            for j in range(P_mesh3.shape[1]):
                _, dP_dt, _, dcatI_dt = compute_derivatives(
                    A0*0.3, P_mesh3[i, j], 0, catI_mesh2[i, j], k_dict)
                dP3[i, j] = dP_dt
                dcatI2[i, j] = dcatI_dt
        
        magnitude5 = np.sqrt(dP3**2 + dcatI2**2)
        magnitude5[magnitude5 == 0] = 1
        
        ax5.quiver(P_mesh3, catI_mesh2, dP3/magnitude5, dcatI2/magnitude5, 
                  magnitude5, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax5.plot(P_sol, catI_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax5.scatter(P_sol[0], catI_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax5.scatter(P_sol[-1], catI_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax5.set_xlabel('[P]', fontsize=10, fontweight='bold')
        ax5.set_ylabel('[catI] (Inhibited)', fontsize=10, fontweight='bold')
        ax5.set_title('P vs catI', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(0, cat_total)
        
        # ========== Plot 6: cat1 vs catI ==========
        ax6.clear()
        cat1_mesh3, catI_mesh3 = np.meshgrid(cat1_range, catI_range)
        dcat1_3 = np.zeros_like(cat1_mesh3)
        dcatI3 = np.zeros_like(catI_mesh3)
        
        for i in range(cat1_mesh3.shape[0]):
            for j in range(cat1_mesh3.shape[1]):
                _, _, dcat1_dt, dcatI_dt = compute_derivatives(
                    A0*0.5, 1.0, cat1_mesh3[i, j], catI_mesh3[i, j], k_dict)
                dcat1_3[i, j] = dcat1_dt
                dcatI3[i, j] = dcatI_dt
        
        magnitude6 = np.sqrt(dcat1_3**2 + dcatI3**2)
        magnitude6[magnitude6 == 0] = 1
        
        ax6.quiver(cat1_mesh3, catI_mesh3, dcat1_3/magnitude6, dcatI3/magnitude6, 
                  magnitude6, cmap='coolwarm', alpha=0.6, scale=25, width=0.004)
        ax6.plot(cat1_sol, catI_sol, 'b-', alpha=0.9, linewidth=2.5)
        ax6.scatter(cat1_sol[0], catI_sol[0], color='green', s=100, 
                   marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax6.scatter(cat1_sol[-1], catI_sol[-1], color='red', s=100, 
                   marker='X', edgecolors='black', linewidth=2, zorder=5)
        
        ax6.set_xlabel('[cat1]', fontsize=10, fontweight='bold')
        ax6.set_ylabel('[catI] (Inhibited)', fontsize=10, fontweight='bold')
        ax6.set_title('cat1 vs catI', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, cat_total)
        ax6.set_ylim(0, cat_total)
        
        # Build title with all rate constants
        k_str = ", ".join([f"k{i}={k_dict.get(f'k{i}', 0.0):.3f}" for i in range(1, num_reactions + 1)])
        plt.suptitle(f'Phase Portrait Analysis | {k_str} | [cat]tot={cat_total:.2f}, [A]₀={A0:.2f}', 
                    fontsize=10, fontweight='bold')
        
        fig.canvas.draw_idle()
    
    # Initial plot with default k-values
    k_init_dict['cat_total'] = cat_total_init
    k_init_dict['A0'] = A0_init
    update_plot(k_init_dict)
    
    # Create sliders - dynamically generate for all reactions
    slider_height = 0.02
    slider_spacing = 0.03
    slider_left = 0.12
    slider_width = 0.76
    
    # Calculate bottom_start based on number of reaction sliders
    # Position sliders well below the plots to avoid overlap
    num_slider_rows = num_reactions + 2  # all reactions + cat_total + A0
    bottom_start = 0.02 + (num_slider_rows * slider_spacing) + slider_spacing
    
    # Create slider axes and sliders for each reaction
    slider_dict = {}
    slider_axes = {}
    
    # Create sliders for all reactions
    for i in range(1, num_reactions + 1):
        row = i - 1
        y_pos = bottom_start - (row * slider_spacing)
        ax = plt.axes((slider_left, y_pos, slider_width, slider_height))
        slider_axes[f'k{i}'] = ax
        
        # Get reaction label from mechanism
        rxn = mech_info['reactions'][i-1]
        reactants = ' + '.join(rxn['reactants'])
        products = ' + '.join(rxn['products'])
        
        slider = Slider(ax, f'k{i}: {reactants}→{products}', 0.0, 3.0, 
                       valinit=k_init_dict[f'k{i}'], valstep=0.01)
        slider_dict[f'k{i}'] = slider
    
    # Create sliders for catalyst and initial reactant
    cat_row = num_reactions
    A0_row = num_reactions + 1
    
    ax_cat = plt.axes((slider_left, bottom_start - (cat_row * slider_spacing), slider_width, slider_height))
    ax_A0 = plt.axes((slider_left, bottom_start - (A0_row * slider_spacing), slider_width, slider_height))
    slider_axes['cat_total'] = ax_cat
    slider_axes['A0'] = ax_A0
    
    slider_cat = Slider(ax_cat, '[cat]total', 0.1, 5.0, valinit=cat_total_init, valstep=0.05)
    slider_A0 = Slider(ax_A0, '[A]₀', 0.1, 10.0, valinit=A0_init, valstep=0.1)
    
    slider_dict['cat_total'] = slider_cat
    slider_dict['A0'] = slider_A0
    
    def update(val):
        # Build k_dict from all sliders
        k_dict = {}
        for i in range(1, num_reactions + 1):
            k_dict[f'k{i}'] = slider_dict[f'k{i}'].val
        k_dict['cat_total'] = slider_dict['cat_total'].val
        k_dict['A0'] = slider_dict['A0'].val
        # Update plot with new rate constants
        update_plot(k_dict)
    
    # Connect all sliders to the update function
    for key, slider in slider_dict.items():
        slider.on_changed(update)
    
    plt.show()


# Run only the interactive version
if __name__ == "__main__":
    # Define the reaction mechanism (8 explicit reactions)
    mechanism = [
        "A + cat -> cat1",           # R1: forward
        "cat1 -> A + cat",           # R2: reverse of R1
        "cat1 -> cat + P",           # R3: decomposition
        "P + cat -> cat1",           # R4: reverse of R3
        "cat + P -> catI",           # R5: inhibition
        "catI -> cat + P",           # R6: reverse of R5
        "A + cat -> catI",           # R7: direct inhibition
        "catI -> A + cat"            # R8: reverse of R7
    ]
    
    print("Interactive 2D Phase Portrait Analysis with 8 Reactions")
    print("=" * 70)
    print("\nReaction Mechanism:")
    print("  R1: A + cat → cat1       (forward)")
    print("  R2: cat1 → A + cat       (reverse)")
    print("  R3: cat1 → cat + P       (decomposition)")
    print("  R4: P + cat → cat1       (reverse)")
    print("  R5: cat + P → catI       (inhibition)")
    print("  R6: catI → cat + P       (reverse)")
    print("  R7: A + cat → catI       (direct inhibition)")
    print("  R8: catI → A + cat       (reverse)")
    print("\nAdjust sliders to explore parameter space:")
    print("  • k₁-k₈: Rate constants for each reaction")
    print("  • [cat]total: Total catalyst concentration")
    print("  • [A]₀: Initial reactant concentration")
    print("\nNote: [cat1]₀, [catI]₀, and [P]₀ are always 0")
    print("=" * 70)
    interactive_2d_portraits(mechanism)
