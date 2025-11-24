import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
import json
from math import log10 as log
import shutil
import multiprocessing as mp
from functools import partial

from chem_sim.simulator import Simulator
from chem_base.reaction import TEReaction
from chem_utils.chem_plot_utils import apply_acs_layout
from chem_utils.chem_logger import chem_logger
chem_logger.setLevel(level=logging.INFO)
import random


def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' ran in {elapsed_time:.4f} seconds.")
        return result
    return wrapper

def _json_safe(o):
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.floating, _np.integer, _np.bool_)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_pd.Timestamp, _pd.Timedelta)):
        return str(o)
    try:
        json.dumps(o)
        return o
    except Exception:
        return str(o)


def _custom_round_static(val):
    """Static version of custom_round for multiprocessing"""
    if pd.isna(val) or val == 0:
        return val
    val = float(val)
    try:
        rounding_digit = abs(2 - int(log(val)))
    except:
        rounding_digit = 10
    rounding_digit += 1
    if val >= 100:
        rounding_digit = 0
    if val < 1e-3:
        val = 0
    return round(val, rounding_digit)

def _custom_round_dataframe(df):
    """Applies custom rounding to a dataframe, with special rules for cat and catI."""
    rounded_df = df.copy()
    for col in rounded_df.columns:
        if col in ['cat', 'catI']:
            rounded_df[col] = rounded_df[col].apply(lambda x: round(x, 6))
        else:
            rounded_df[col] = rounded_df[col].apply(_custom_round_static)
    return rounded_df


def _get_used_reagents_static(reactions):
    """Static version for multiprocessing"""
    total_sm = []
    for rct in reactions:
        for chem in (rct.educts + rct.products):
            # Exclude catalyst-related species that aren't starting materials from this list
            if "cat" not in chem.label and "P" not in chem.label:
                total_sm.append(str(chem))
    return sorted(list(set(total_sm)))


def _get_random_k_values_static(num_k_values, activation_info=None):
    """
    Static version for multiprocessing.
    Handles special k-value constraints for activation mechanisms.
    
    For activation mechanisms:
    - k1 (activation step) is 0.5% to 1.5% of the RDS from non-activation steps (k2+)
    - The RDS is identified as the step with the smallest k value among non-activation steps
    """
    k_dict = {}
    
    if activation_info and activation_info['is_activation']:
        # Special logic for activation mechanisms
        # Step 1 is activation (cat -> cat1)
        # Steps 2+ are the main reaction pathway
        
        # 1. Generate k values for steps 2 and higher (non-activation steps)
        non_activation_k_values = []
        for i in range(1, num_k_values):  # i.e., for k2, k3, k4, ...
            k_val = np.round(np.random.uniform(0, 1) * np.random.choice([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]), 4)
            non_activation_k_values.append(k_val)
            
            key_k = f"k{i+1}"
            key_kN = f"kN{i+1}"
            k_dict[key_k] = k_val
            k_dict[key_kN] = np.round(k_val / np.random.uniform(0.8, 20), 3)
        
        # 2. Identify RDS (smallest k value among non-activation steps)
        rds_k_value = min(non_activation_k_values)
        
        # 3. Generate k1 as 0.5% to 1.5% of the RDS
        k1_factor = np.random.uniform(0.005, 0.015)  # 0.5% to 1.5%
        k1_val = np.round(rds_k_value * k1_factor, 4)
        k_dict['k1'] = k1_val
        k_dict['kN1'] = np.round(k1_val / np.random.uniform(0.8, 20), 3)

    else:
        # Original logic for non-activation mechanisms
        random_k_values = np.random.uniform(0, 1, num_k_values)
        random_k_factors = np.random.choice([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], num_k_values)
        rounded_k_values = np.round(random_k_values * random_k_factors, 4)
        
        for i in range(num_k_values):
            key_k = f"k{i+1}"
            key_kN = f"kN{i+1}"
            random_kN_factor = np.random.uniform(0.8, 20)
            k_dict[key_k] = rounded_k_values[i]
            k_dict[key_kN] = np.round(rounded_k_values[i] / random_kN_factor, 3)
    
    return k_dict


def _get_random_c_values_static(num_c_values, activation_info=None):
    """
    Static version for multiprocessing.
    Returns (c_values_list, cat_or_catx_conc)
    """
    lower_bound, upper_bound = 0.05, 1.0
    first_c_value = round(np.random.uniform(lower_bound, upper_bound), 3)
    c_factors = np.random.uniform(0.333, 3, num_c_values - 1)
    c_factors[(c_factors >= 0.95) & (c_factors <= 1.05)] = 1
    
    c_values = np.empty(num_c_values)
    c_values[0] = first_c_value
    for i in range(1, num_c_values):
        c_values[i] = round(c_values[i - 1] * c_factors[i - 1], 3)
    
    lower_cat_bound, upper_cat_bound = 0.01, 0.1
    cat_factor = np.random.uniform(lower_cat_bound, upper_cat_bound)
    
    # The concentration is for the catalyst precursor (e.g., 'cat' or 'catx')
    cat_conc = float(round(np.min(c_values) * cat_factor, 3))
    
    return c_values.tolist(), cat_conc


def _apply_noise_static(df, noise_pct, noise_type):
    """Static version for multiprocessing"""
    # Don't apply noise to time, cat, or catI columns
    exclude_cols = ['time', 'cat', 'catI']
    df_noisy = df.loc[:, ~df.columns.isin(exclude_cols)].copy()

    if noise_type == "gauss":
        std = noise_pct / 3
        noise_factor = np.random.normal(1, std, size=df_noisy.shape)
        df_noisy *= noise_factor

    elif noise_type == "uniform":
        lower_bound = 1 - noise_pct
        upper_bound = 1 + noise_pct
        noise_factor = np.random.uniform(lower_bound, upper_bound, size=df_noisy.shape)
        df_noisy *= noise_factor

    elif noise_type == "realistic":
        lower_bound = 1 - noise_pct
        upper_bound = 1 + (noise_pct/4)
        noise_factor = np.random.uniform(lower_bound, upper_bound, size=df_noisy.shape)
        df_noisy *= noise_factor
        std = noise_pct / 3
        noise_factor = np.random.normal(1, std, size=df_noisy.shape)
        df_noisy *= noise_factor

    # Add back the excluded columns without noise
    df_noisy["time"] = df["time"].copy() 
    if "cat" in df.columns:
        df_noisy["cat"] = df["cat"].copy()
    if "catI" in df.columns:
        df_noisy["catI"] = df["catI"].copy()
    return df_noisy


def _check_catalyst_deactivation_at_50_pct_product(df, initial_cat_conc):
    """
    New deactivation check: catI must be >= 10% of initial catalyst
    concentration at the timepoint where 50% of the final product has formed.
    """
    if "P" not in df.columns or "catI" not in df.columns or df.empty:
        return False, 0.0

    final_product_conc = df["P"].iloc[-1]
    if final_product_conc <= 0 or initial_cat_conc <= 0:
        return False, 0.0

    target_product_conc = final_product_conc * 0.5
    
    # Find the index closest to the 50% product mark
    idx_at_50_pct_p = (df['P'] - target_product_conc).abs().idxmin()
    
    # Get the catI concentration at that specific index
    catI_at_50_pct_p = df['catI'].loc[idx_at_50_pct_p]
    
    # Calculate the fraction
    catI_fraction = catI_at_50_pct_p / initial_cat_conc
    
    # Check if the condition is met (>= 10%)
    is_valid = catI_fraction >= 0.1 # set to 0.1 for 10% deactivation
    
    return is_valid, catI_fraction


def _normalize_for_limiting_reagent_static(df, limiting_reagent):
    """Static version for multiprocessing"""
    max_value = df[limiting_reagent].max()
    norm_df = pd.DataFrame()
    for c in df.columns:
        if c == "time":
            continue
        norm_df[c] = df[c] / max_value
    norm_df["time"] = df["time"].copy()
    return norm_df


def _check_convergence_static(p_series, nr_conv_points, trailing_points):
    """Static version for multiprocessing"""
    if p_series is None or len(p_series) < max(3, nr_conv_points + 1):
        return (False, None, None)

    diffs = p_series.diff().abs()

    max_diff = diffs.max()
    if pd.isna(max_diff) or max_diff == 0:
        base_idx = max(0, len(p_series) - (nr_conv_points + 1))
        conv_idx = min(base_idx + trailing_points, len(p_series) - 1)
        nr_conv_points = len(p_series) - conv_idx
        return (True, conv_idx, nr_conv_points)

    diffs = diffs / max_diff

    last_d_avg = float(diffs.tail(nr_conv_points).mean())
    converged = last_d_avg < 0.005

    conv_target = float(p_series.iloc[-1]) * 0.95
    base_idx = int((p_series - conv_target).abs().idxmin())
    conv_idx = base_idx + int(trailing_points)
    conv_idx = max(0, min(conv_idx, len(p_series) - 1))
    nr_conv_points = len(p_series) - conv_idx

    if converged:
        return (True, conv_idx, nr_conv_points)

    return (False, None, None)


def _find_end_time_static(m, k_dict, c_dict, check_deactivation,
                          min_yield, non_limiting_chems, trailing_points,
                          nr_conv_points, conv_tolerance, activation_info=None):
    """Static version for multiprocessing with new deactivation check."""
    num_data_points = 30
    
    # For deactivation check, the reference concentration is the total catalyst added
    initial_cat_conc = c_dict.get("cat", 0)
    if activation_info and activation_info['is_activation']:
        initial_cat_conc = c_dict.get(activation_info['precursor'], 0)

    limiting_reagent = min((c for c in c_dict if c not in non_limiting_chems), key=lambda c: c_dict[c])
    
    sim = Simulator()
    error_tuple = (None, None, None)
    t_stop = 1
    i = 0

    while True:
        i += 1

        if i > 8 and sim.result["P"].max() <= 0:
            return error_tuple
                
        if i > 20:
            return error_tuple
        
        sim.setup(reactions=m, k_dict=k_dict, c_dict=c_dict)
        
        selections = list(c_dict.keys()) + ["time"]
        if any("catI" in str(rxn) for rxn in m):
            if "catI" not in selections:
                selections.append("catI")
        
        # ADD TRY-EXCEPT HERE
        try:
            sim.simulate(0, t_stop, num_data_points, use_const_cat=False, selections=selections)
        except RuntimeError as e:
            if "CVODE" in str(e) or "CV_ERR_FAILURE" in str(e) or "No sbml element exists" in str(e):
                # Numerical integration failed or invalid species - these k values are bad
                return error_tuple
            else:
                raise  # Re-raise if it's a different error

        if sim.result["P"].max() <= 0:
            t_stop *= 2
            continue

        norm_result = _normalize_for_limiting_reagent_static(sim.result, limiting_reagent=limiting_reagent)
        is_converged, conv_index, nr_conv_points_result = _check_convergence_static(
            norm_result["P"], nr_conv_points, trailing_points
        )
        yield_ = round(norm_result["P"].max(), 2)

        if is_converged:
            # When is_converged is True, conv_index is guaranteed to be an int (not None)
            true_t_stop = _custom_round_static(norm_result["time"].iloc[conv_index])  # type: ignore
            
            if yield_ < min_yield:
                return error_tuple
            
            if check_deactivation:
                # Perform a full simulation to the converged time to check deactivation accurately
                full_sim = Simulator()
                full_sim.setup(reactions=m, k_dict=k_dict, c_dict=c_dict)
                full_sim.simulate(0, true_t_stop, 100, use_const_cat=False, selections=selections)

                is_deactivated, _ = _check_catalyst_deactivation_at_50_pct_product(
                    full_sim.result, initial_cat_conc
                )
                if not is_deactivated:
                    return error_tuple
            
            return limiting_reagent, true_t_stop, yield_

        else:
            t_stop *= 2


def _process_single_reaction(args):
    """
    Worker function to process a single reaction in parallel.
    """
    (reaction_idx, m_name, m, concs_dict, save_folder, 
     nr_data_points_bounds, max_k_attempts, noise_pct, noise_type,
     require_deactivation, mechanism_has_deactivation,
     mechanism_has_activation,
     min_yield, non_limiting_chems, trailing_points, 
     nr_conv_points, conv_tolerance) = args
    
    should_check_deactivation = require_deactivation and mechanism_has_deactivation.get(m_name, False)
    activation_info = mechanism_has_activation.get(m_name) # Contains is_activation and precursor
    
    # Get used reagents
    used_sm = _get_used_reagents_static(m)
    
    # Try multiple k values until we get valid deactivation (if required)
    k_attempts = 0
    valid_k_found = False
    
    while k_attempts < max_k_attempts:
        k_attempts += 1
        
        # Get random k and c values
        k_dict = _get_random_k_values_static(num_k_values=len(m), activation_info=activation_info)
        c_list, cat_precursor_conc = _get_random_c_values_static(len(used_sm), activation_info=activation_info)
        
        c_dict = {species: c for species, c in zip(used_sm, c_list)}
        if activation_info['is_activation']:
            c_dict[activation_info['precursor']] = cat_precursor_conc
            # Dynamically find all catalyst species from reactions and set them to 0
            all_species = set()
            for r in m:
                all_species.update(str(s) for s in r.educts)
                all_species.update(str(s) for s in r.products)
            
            for species in all_species:
                if 'cat' in species and species != activation_info['precursor']:
                    c_dict[species] = 0
        else:
            c_dict["cat"] = cat_precursor_conc

        c_dict["P"] = 0
        if "I" in c_dict.keys():
            c_dict["I"] = 0
        if any("catI" in str(rxn) for rxn in m):
            c_dict["catI"] = 0
        
        # Apply first experiment concentrations
        c_instr_dict = concs_dict[1]
        
        if c_instr_dict:
            for s_str, val in c_instr_dict.items():
                species, info = s_str.split("_")
                if species not in c_dict.keys():
                    continue
                if info == "conc":
                    c_dict[species] = val
        
        # Test if this k_dict gives valid results
        limiting_reagent, true_t_stop, yield_ = _find_end_time_static(
            m=m, c_dict=c_dict, k_dict=k_dict, 
            check_deactivation=should_check_deactivation,
            min_yield=min_yield,
            non_limiting_chems=non_limiting_chems,
            trailing_points=trailing_points,
            nr_conv_points=nr_conv_points,
            conv_tolerance=conv_tolerance,
            activation_info=activation_info
        )
        
        if true_t_stop:  # Valid result found
            valid_k_found = True
            break
    
    if not valid_k_found:
        return None
    
    # Create reaction folder
    mech_path = f"{save_folder}/{m_name}/rct_{reaction_idx}"
    os.makedirs(mech_path, exist_ok=True)
    
    info_dict = {}
    info_dict["noise_pct"] = noise_pct
    info_dict["noise_type"] = noise_type
    info_dict["require_deactivation"] = require_deactivation
    info_dict["min_catI_fraction_at_50_pct_P"] = 0.1 # set to 0.1 for 10% deactivation
    info_dict["require_activation"] = activation_info['is_activation']
    info_dict["mechanism"] = [str(rct).split(": ")[1] for rct in m]
    info_dict["k_dict"] = k_dict
    info_dict["k_attempts"] = k_attempts
    
    # Select number of data points ONCE for all experiments
    num_data_points = int(np.random.uniform(
        low=nr_data_points_bounds[0], 
        high=nr_data_points_bounds[1]
    ))
    info_dict["num_data_points"] = num_data_points
    
    csv_paths = []
    base_c_dict: dict | None = None
    
    # Track deactivation diagnostics
    catI_fractions_at_50_pct = {}
    
    # Process all experiments
    for i in range(1, len(concs_dict) + 1):
        c_instr_dict = concs_dict[i]
        
        # All random
        if not c_instr_dict:
            c_list, cat_precursor_conc = _get_random_c_values_static(len(used_sm), activation_info=activation_info)
            c_dict = {species: c for species, c in zip(used_sm, c_list)}
            if activation_info['is_activation']:
                c_dict[activation_info['precursor']] = cat_precursor_conc
                all_species = set()
                for r in m:
                    all_species.update(str(s) for s in r.educts)
                    all_species.update(str(s) for s in r.products)
                for species in all_species:
                    if 'cat' in species and species != activation_info['precursor']:
                        c_dict[species] = 0
            else:
                c_dict["cat"] = cat_precursor_conc
            c_dict["P"] = 0
            if "I" in c_dict.keys():
                c_dict["I"] = 0
            if any("catI" in str(rxn) for rxn in m):
                c_dict["catI"] = 0
        
        # Update concentrations for this experiment
        if i != 1 and base_c_dict is not None:
            c_dict = base_c_dict.copy()
            # Reset catI to 0 for each new experiment
            if any("catI" in str(rxn) for rxn in m):
                c_dict["catI"] = 0
        
        for s_str, val in c_instr_dict.items():
            species, info = s_str.split("_")
            
            if species not in c_dict.keys():
                continue
            
            if info == "conc":
                new_conc = val
                
            elif info == "fact":
                if i == 1:
                    continue
                if base_c_dict is None:
                    continue
                if isinstance(val, dict):
                    used_fact = np.random.uniform(val["lower_bound"], val["upper_bound"])
                    new_conc = base_c_dict[species] * used_fact
                elif isinstance(val, tuple):
                    used_fact = np.random.uniform(val[0], val[1])
                    new_conc = base_c_dict[species] * used_fact
                else:
                    new_conc = base_c_dict[species] * val
            
            c_dict[species] = new_conc
        
        if i == 1:
            base_c_dict = c_dict.copy()
        
        info_dict[f"c_dict_{i}"] = c_dict
        
        # Find end time (only check deactivation on first experiment if requested)
        limiting_reagent, true_t_stop, yield_ = _find_end_time_static(
            m=m, c_dict=c_dict, k_dict=k_dict,
            check_deactivation=(i == 1 and should_check_deactivation),
            min_yield=min_yield,
            non_limiting_chems=non_limiting_chems,
            trailing_points=trailing_points,
            nr_conv_points=nr_conv_points,
            conv_tolerance=conv_tolerance,
            activation_info=activation_info
        )
        
        # Ensure first experiment converges
        if not true_t_stop and i == 1:
            try:
                shutil.rmtree(mech_path)
            except Exception:
                pass
            return None
        
        if not true_t_stop and i > 1:
            true_t_stop = 2**20
        
        sim = Simulator()
        sim.setup(reactions=m, k_dict=k_dict, c_dict=c_dict)
        
        selections = ["time"] + list(c_dict.keys())
        if should_check_deactivation:
            if "catI" not in selections:
                selections.append("catI")
        
        # ADD ERROR HANDLING FOR CVODE ERRORS
        try:
            sim.simulate(0, true_t_stop, num_data_points, use_const_cat=False, selections=selections)
        except RuntimeError as e:
            if "CVODE" in str(e) or "CV_ERR_FAILURE" in str(e) or "No sbml element exists" in str(e):
                try:
                    shutil.rmtree(mech_path)
                except Exception:
                    pass
                return None
            else:
                raise
        
        # Deactivation fraction check
        if should_check_deactivation:
            initial_cat_total = 0
            if activation_info['is_activation']:
                initial_cat_total = float(c_dict.get(activation_info['precursor'], 0))
            else:
                initial_cat_total = float(c_dict.get("cat", 0))

            is_deactivated, catI_frac = _check_catalyst_deactivation_at_50_pct_product(sim.result, initial_cat_total)
            info_dict[f"catI_fraction_at_50_pct_P_{i}"] = catI_frac
            catI_fractions_at_50_pct[i] = catI_frac

        info_dict[f"num_data_points_{i}"] = num_data_points
        info_dict[f"true_t_stop_{i}"] = true_t_stop
        info_dict[f"limiting_reagent_{i}"] = limiting_reagent
        info_dict[f"yield_{i}"] = yield_
        
        # Apply noise and save
        final_df_noisy = _apply_noise_static(sim.result.copy(), noise_pct, noise_type)
        final_df = _custom_round_dataframe(final_df_noisy)
        
        if len(final_df) > 1:
            csv_path = f"{mech_path}/exp_{i-1}.csv"
            final_df.to_csv(csv_path, index=False)
            csv_paths.append(csv_path)
        else:
            try:
                shutil.rmtree(mech_path)
            except Exception:
                pass
            return None
    
    # VALIDATE BOTH EXPERIMENTS if deactivation is required
    if should_check_deactivation:
        exp1_valid = catI_fractions_at_50_pct.get(1, 0) >= 0.1 # Change to 0.1 for 10% deactivation
        exp2_valid = catI_fractions_at_50_pct.get(2, 0) >= 0.1 # Change to 0.1 for 10% deactivation
        
        if not (exp1_valid and exp2_valid):
            try:
                shutil.rmtree(mech_path)
            except Exception:
                pass
            return None
    
    # Write info dict
    info_dict["csv_paths"] = csv_paths
    info_dict_path = os.path.join(mech_path, "info_dict.json")
    try:
        with open(info_dict_path, "w") as file:
            json.dump(info_dict, file, indent=4, default=_json_safe)
    except Exception as e:
        logging.exception(f"Failed to write {info_dict_path}: {e}")
    
    return csv_paths, k_dict


# TimeCourseCreator
class TimeCourseCreator():
    def __init__(self, mechanisms, min_yield=0.5, noise_pct=0.03, noise_type="realistic", 
                 trailing_points=3, require_deactivation=False,
                 require_activation=False): # Removed unused parameters
        self.NON_LIMITING_CHEMS = {"P", "time", "L", "L*", "I", "catI", 'P1', 'P2'}
    
        self.MIN_YIELD = min_yield

        self.NOISE_PCT = noise_pct
        self.NOISE_TYPE = noise_type
        self.TRAILING_POINTS = trailing_points
        
        # Catalyst deactivation parameters
        self.REQUIRE_DEACTIVATION = require_deactivation
        
        # Catalyst activation parameters
        self.REQUIRE_ACTIVATION = require_activation
        
        # dont change unless you are sure what they do
        self.NR_CONV_POINTS = 3
        self.CONV_TOLERANCE = 0.01

        if isinstance(mechanisms, list):
            mechanisms = {"default_M": mechanisms}

        # Auto-detect deactivation and activation
        self.mechanism_has_deactivation = {}
        self.mechanism_has_activation = {}
        
        for mechanism_name, reactions in mechanisms.items():
            # Check for deactivation (presence of catI or I)
            has_deactivation = any("catI" in rxn or "-> I" in rxn for rxn in reactions)
            self.mechanism_has_deactivation[mechanism_name] = has_deactivation
            
            # Check for activation (e.g., "cat -> cat1")
            first_reaction = reactions[0]
            parts = [p.strip() for p in first_reaction.split("->")]
            is_activation = (len(parts) == 2 and parts[0] == "cat" and parts[1] == "cat1" and "+" not in parts[0])
            
            precursor = None
            if is_activation:
                precursor = parts[0]
                # Add all catalyst species to non-limiting to be safe
                self.NON_LIMITING_CHEMS.add("cat")
                self.NON_LIMITING_CHEMS.add("cat1")
                self.NON_LIMITING_CHEMS.add("cat2")


            self.mechanism_has_activation[mechanism_name] = {
                "is_activation": is_activation,
                "precursor": precursor
            }
            
            print(f"\nMechanism: {mechanism_name}")
            for rxn in reactions:
                print(f"  - {rxn}")
            
            if is_activation:
                print(f"  → Detected catalyst activation (precursor: {precursor})")
            if has_deactivation:
                print(f"  → Detected catalyst deactivation (catI or I found)")
            
            if not is_activation and not has_deactivation:
                print(f"  → Standard mechanism")

            mechanisms[mechanism_name] = self.convert_reactions(reactions)

        self.mechanisms = mechanisms
    
    def convert_reactions(self, reactions):
        reactions = [TEReaction(reaction_string=rct, id_=i+1) for i, rct in enumerate(reactions)]
        return reactions
    
    def create_dataset(self, num_reactions_per_mechanism, concs_dict, save_folder=None,
                      nr_data_points_bounds=(15, 40), max_k_attempts=50, n_processes=None,
                      use_parallel=True):
        """
        Create dataset with optional parallelization.
        """
        
        if use_parallel:
            return self._create_dataset_parallel(
                num_reactions_per_mechanism, concs_dict, save_folder,
                nr_data_points_bounds, max_k_attempts, n_processes
            )
        else:
            # Serial mode is deprecated and has been removed for simplicity.
            # Please use parallel mode.
            raise NotImplementedError("Serial mode is deprecated. Please use `use_parallel=True`.")
    
    def _create_dataset_parallel(self, num_reactions_per_mechanism, concs_dict, 
                                save_folder, nr_data_points_bounds, max_k_attempts, 
                                n_processes):
        """Parallelized version of create_dataset"""
        
        if n_processes is None:
            n_processes = max(1, mp.cpu_count() - 1)
        
        print(f"\n{'='*60}")
        print(f"PARALLEL MODE: Using {n_processes} processes")
        print(f"{'='*60}\n")
        
        start_time = time.perf_counter()
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
        
        all_results = []
        
        for m_name, m in self.mechanisms.items():
            should_check_deactivation = self.REQUIRE_DEACTIVATION and self.mechanism_has_deactivation.get(m_name, False)
            activation_info = self.mechanism_has_activation.get(m_name, {"is_activation": False, "precursor": None})

            print(f"Processing {m_name} with {num_reactions_per_mechanism} reactions")
            print(f"  - REQUIRE_DEACTIVATION: {self.REQUIRE_DEACTIVATION}")
            print(f"  - mechanism_has_deactivation: {self.mechanism_has_deactivation.get(m_name, False)}")
            print(f"  - should_check_deactivation: {should_check_deactivation}")
            print(f"  - mechanism_has_activation: {activation_info['is_activation']}")
            
            if activation_info['is_activation']:
                print(f"  → WITH activation k-value constraints (k1 vs k2)")
            if should_check_deactivation:
                print(f"  → WITH deactivation validation (catI >= 10% of initial cat at 50% P)")
                print(f"  → BOTH experiments must meet threshold")

            # Prepare arguments for parallel processing
            args_list = [
                (idx, m_name, m, concs_dict, save_folder, nr_data_points_bounds,
                 max_k_attempts, self.NOISE_PCT, self.NOISE_TYPE,
                 self.REQUIRE_DEACTIVATION, self.mechanism_has_deactivation,
                 self.mechanism_has_activation,
                 self.MIN_YIELD, self.NON_LIMITING_CHEMS,
                 self.TRAILING_POINTS, self.NR_CONV_POINTS, self.CONV_TOLERANCE)
                for idx in range(num_reactions_per_mechanism)
            ]
            
            # Run in parallel
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(_process_single_reaction, args_list)
            
            # Filter successful results
            successful = [r for r in results if r is not None]
            all_results.extend(successful)
            
            print(f"\n✓ Successfully generated {len(successful)}/{num_reactions_per_mechanism} reactions for {m_name}")
        
        elapsed = time.perf_counter() - start_time
        print(f"\n{'='*60}")
        print(f"COMPLETED in {elapsed:.2f}s ({elapsed/60:.1f} minutes)")
        print(f"Total successful reactions: {len(all_results)}")
        print(f"{'='*60}\n")
        
        if all_results:
            return all_results[0] if len(all_results) == 1 else all_results
        return None

def merge_batch_folders(batch_folders, output_folder):
    """
    Merge multiple batch folders into a single output folder with renumbered reactions.
    
    Args:
        batch_folders: List of folder paths to merge (e.g., ["batch0", "batch1", "batch2"])
        output_folder: Path to the merged output folder
    """
    import os
    import shutil
    import json
    from pathlib import Path
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Track the current reaction number for each mechanism
    mechanism_counters = {}
    total_reactions_copied = 0
    
    for batch_folder in batch_folders:
        if not os.path.exists(batch_folder):
            print(f"Warning: {batch_folder} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {batch_folder}...")
        
        # Iterate through mechanism folders (e.g., M1_n, M2_cd, etc.)
        for mechanism_name in os.listdir(batch_folder):
            mechanism_path = os.path.join(batch_folder, mechanism_name)
            
            if not os.path.isdir(mechanism_path):
                continue
            
            # Initialize counter for this mechanism if not exists
            if mechanism_name not in mechanism_counters:
                mechanism_counters[mechanism_name] = 0
            
            # Create mechanism folder in output if it doesn't exist
            output_mechanism_path = os.path.join(output_folder, mechanism_name)
            os.makedirs(output_mechanism_path, exist_ok=True)
            
            # Get all reaction folders (rct_0, rct_1, etc.)
            reaction_folders = [f for f in os.listdir(mechanism_path) if f.startswith('rct_')]
            
            for reaction_folder in sorted(reaction_folders):
                source_rct_path = os.path.join(mechanism_path, reaction_folder)
                
                if not os.path.isdir(source_rct_path):
                    continue
                
                # Create new numbered folder in output
                new_rct_num = mechanism_counters[mechanism_name]
                dest_rct_path = os.path.join(output_mechanism_path, f"rct_{new_rct_num}")
                
                # Copy the entire reaction folder
                shutil.copytree(source_rct_path, dest_rct_path)
                
                # Update the info_dict.json if it exists to reflect new paths
                info_dict_path = os.path.join(dest_rct_path, "info_dict.json")
                if os.path.exists(info_dict_path):
                    try:
                        with open(info_dict_path, 'r') as f:
                            info_dict = json.load(f)
                        
                        # Update csv_paths to reflect new directory structure
                        if "csv_paths" in info_dict:
                            new_csv_paths = []
                            for old_path in info_dict["csv_paths"]:
                                # Extract just the filename
                                filename = os.path.basename(old_path)
                                new_path = os.path.join(dest_rct_path, filename)
                                new_csv_paths.append(str(Path(new_path))) # Use Path for robust paths
                            info_dict["csv_paths"] = new_csv_paths
                        
                        # Write updated info_dict
                        with open(info_dict_path, 'w') as f:
                            json.dump(info_dict, f, indent=4, default=_json_safe)
                    except Exception as e:
                        print(f"Warning: Could not update info_dict.json in {dest_rct_path}: {e}")
                
                mechanism_counters[mechanism_name] += 1
                total_reactions_copied += 1
                
                if total_reactions_copied % 100 == 0:
                    print(f"  Copied {total_reactions_copied} reactions so far...")
    
    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Total reactions copied: {total_reactions_copied}")
    print(f"\nReactions per mechanism:")
    for mechanism, count in sorted(mechanism_counters.items()):
        print(f"  {mechanism}: {count} reactions")
    print(f"\nOutput folder: {output_folder}")
    
    return mechanism_counters



if __name__ == "__main__":
    mechanisms = {
        # Standard mechanisms
        "M1_n": ["A + cat -> cat1", "cat1 -> cat + P"],
        "M1_cd": ["A + cat -> cat1", "cat1 -> cat + P", "cat -> catI"],
        "M1_scd": ["A + cat -> cat1", "cat1 -> cat + P", "S + cat -> catI"],
        "M1_pcd": ["A + cat -> cat1", "cat1 -> cat + P", "cat + P -> catI"],
        "M1_c1d": ["A + cat -> cat1", "cat1 -> cat + P", "cat1 -> catI"],
        "M1_pc1d": ["A + cat -> cat1", "cat1 -> cat + P", "cat1 + P -> catI"],
        "M1_sc1d": ["A + cat -> cat1", "cat1 -> cat + P", "A + cat1 -> catI"],
        # # "M1_dd": ["A + cat -> cat1", "cat1 -> cat + P", "cat + cat -> catI"],
        "M2_n": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P'],
        "M2_cd": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat -> catI'],
        "M2_scd": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'A + cat -> catI'],
        "M2_pcd": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat + P -> catI'],
        "M2_c1d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat1 -> catI'],
        "M2_pc1d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat1 + P -> catI'],
        "M2_sc1d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'A + cat1 -> catI'],
        "M2_c2d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat2 -> catI'],
        "M2_pc2d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat2 + P -> catI'],
        "M2_sc2d": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'A + cat2 -> catI'],
        # # "M2_dd": ['A + cat -> cat1', 'cat1 + B -> cat2', 'cat2 -> cat + P', 'cat + cat -> catI'],
        "M3_n": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P'],
        "M3_cd": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat -> catI'],
        "M3_scd": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'A + cat -> catI'],
        "M3_pcd": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat + P -> catI'],
        "M3_c1d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat1 -> catI'],
        "M3_pc1d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat1 + P -> catI'],
        "M3_sc1d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'A + cat1 -> catI'],
        "M3_c2d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat2 -> catI'],
        "M3_pc2d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat2 + P -> catI'],
        "M3_sc2d": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'A + cat2 -> catI'],
        # # "M3_dd": ['A + cat -> cat1', 'cat1 -> cat + P', 'cat1 + A -> cat2', 'cat2 -> cat1 + P', 'cat + cat -> catI'],

        # # New activation mechanisms
        # "M1a_n": ["cat -> cat1", "A + cat1 -> cat2", "cat2 -> cat1 + P"],
        # "M1a_cd": ["cat -> cat1", "A + cat1 -> cat2", "cat2 -> cat1 + P", "cat -> catI"],
        # "M1a_scd": ["cat -> cat1", "A + cat1 -> cat2", "cat2 -> cat1 + P", "S + cat -> catI"],
        # "M1a_pcd": ["cat -> cat1", "A + cat1 -> cat2", "cat2 -> cat1 + P", "cat + P -> catI"],
        # # # "M1_a_dd": ["cat -> cat1", "A + cat1 -> cat2", "cat2 -> cat1 + P", "cat + cat -> catI"],
        # "M2a_n": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 + B -> cat3','cat3 -> cat1 + P'],
        # "M2a_cd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 + B -> cat3','cat3 -> cat1 + P', 'cat -> catI'],
        # "M2a_scd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 + B -> cat3', 'cat3 -> cat1 + P', 'S + cat -> catI'],
        # "M2a_pcd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 + B -> cat3', 'cat3 -> cat1 + P', 'cat + P -> catI'],
        # # # "M2a_dd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 + B -> cat3', 'cat3 -> cat1 + P', 'cat + cat -> catI'],
        # "M3a_n": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 -> cat1 + P', 'cat2 + A -> cat3', 'cat3 -> cat2 + P'],
        # "M3a_cd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 -> cat1 + P', 'cat2 + A -> cat3', 'cat3 -> cat2 + P', 'cat -> catI'],
        # "M3a_scd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 -> cat1 + P', 'cat2 + A -> cat3', 'cat3 -> cat2 + P', 'S + cat -> catI'],
        # "M3a_pcd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 -> cat1 + P', 'cat2 + A -> cat3', 'cat3 -> cat2 + P', 'cat + P -> catI'],
        # # "M3a_dd": ['cat -> cat1', 'A + cat1 -> cat2', 'cat2 -> cat1 + P', 'cat2 + A -> cat3', 'cat3 -> cat2 + P', 'cat + cat -> catI'], 


    }
    Names_of_mechanisms = list(mechanisms.keys())
    print("Mechanisms to be simulated:", Names_of_mechanisms)
    # Initialize the TimeCourseCreator with the test mechanisms
    fake_reactor = TimeCourseCreator(
        mechanisms=mechanisms,
        require_deactivation=True, 
        noise_pct=0.02, 
        noise_type="realistic", 
        trailing_points=3,
        min_yield=0.5
    )
    
    # Define concentration modifications for the experiments
    concs_mod_dict = {  
        1: {"A_conc": 0.1, "cat_conc": 0.002, "B_conc":0.11}, # Base experiment
        2: {"cat_fact": {"lower_bound": 0.25, "upper_bound": 0.75}}, # Second experiment
    }

    for batch in range(1):
        fake_reactor.create_dataset(
            num_reactions_per_mechanism=50,
            concs_dict=concs_mod_dict,
            save_folder=f"M1M2M3_data_{batch}",
            nr_data_points_bounds=(100,100),
            max_k_attempts=500, 
            n_processes=None,  
            use_parallel=True  
        )

    merge_batch_folders(
        batch_folders=[f"M1M2M3_data_{batch}" for batch in range(1)],     
        output_folder="M1M2M3_data_Merged"
    )
    