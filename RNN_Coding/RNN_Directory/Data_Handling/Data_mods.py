"""This will be a data modification file, it will combine a few different modules that I have put together to clean up my filespace.
Currently this has the normalizer, VTNA data labeler, """
import os
import re
import pandas as pd
import numpy as np

def _sanitize(name: str) -> str:
    # lower, strip, remove brackets/punct/spaces -> e.g., "[A ]" -> "a"
    s = re.sub(r"[\[\]\(\)\{\}\-_/\\\s]", "", str(name).lower())
    return s

def _pick_AP_by_header(df: pd.DataFrame):
    """
    Try to find 'A' and 'P' by header names (case-insensitive, tolerant of brackets).
    Returns a 2-col DataFrame with columns ['A','P'] or None if not found.
    """
    colmap = {c: _sanitize(c) for c in df.columns}
    # exact matches after sanitize
    a_candidates = [c for c, s in colmap.items() if s == "a"]
    p_candidates = [c for c, s in colmap.items() if s == "p"]

    if a_candidates and p_candidates:
        a_col = a_candidates[0]
        p_col = p_candidates[0]
        out = df[[a_col, p_col]].copy()
        out.columns = ["A", "P"]
        return out

    # fallback: columns that start with 'a' or 'p' (e.g., a_norm, p_raw)
    a_candidates = [c for c, s in colmap.items() if s.startswith("a")]
    p_candidates = [c for c, s in colmap.items() if s.startswith("p")]
    if a_candidates and p_candidates:
        a_col = a_candidates[0]
        p_col = p_candidates[0]
        out = df[[a_col, p_col]].copy()
        out.columns = ["A", "P"]
        return out

    return None

def _pick_AP_by_position(df: pd.DataFrame):
    """
    If headers don't help, pick the first two numeric columns as A then P.
    NOTE: This assumes A is the first numeric column and P the second.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    out = num_df.iloc[:, :2].copy()
    out.columns = ["A", "P"]
    return out

def _extract_AP_anyhow(csv_path: str):
    """
    Try multiple read strategies, then pick A/P by header or by position.
    Returns df with ['A','P'] or raises ValueError.
    """
    candidates = []
    # Try a few parse styles (header vs no header, skip first row)
    try:
        candidates.append(pd.read_csv(csv_path))  # header=0
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(csv_path, header=None))  # no header
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(csv_path, skiprows=1))  # skip 1, header=0
    except Exception:
        pass
    try:
        candidates.append(pd.read_csv(csv_path, skiprows=1, header=None))  # skip 1, no header
    except Exception:
        pass

    # Attempt extraction on each candidate
    for df in candidates:
        if df is None or df.empty:
            continue
        # First: header-based detection
        ap = _pick_AP_by_header(df)
        if ap is not None:
            return ap
        # Second: numeric position-based detection
        ap = _pick_AP_by_position(df)
        if ap is not None:
            return ap

    raise ValueError("Could not detect A and P columns via header or numeric positions.")

def normalize_and_save_to_training(start_iteration, end_iteration, mechanism_prefixes,
                                   source_root="/Users/dylanpyle/VsCode/test",
                                   dest_root="/Users/dylanpyle/VsCode/RNN_Code/DATA/training",
                                   exps=(0,1)):
    """
    Reads CSVs, auto-detects columns A and P (any header order or no header),
    normalizes by max(A), ensures order A then P, and saves 2-column CSVs
    (no header) at the destination path.
    """
    for i in range(start_iteration, end_iteration + 1):
        for j in exps:
            for prefix in mechanism_prefixes:
                src = f"{source_root}/{prefix}/rct_{i}/exp_{j}.csv"
                dst = f"{dest_root}/{prefix}/rct_{i}/exp_{j}.csv"

                if not os.path.exists(src):
                    print(f"File not found: {src}. Skipping.")
                    continue

                try:
                    # full_df = pd.read_csv(src)
                    df = _extract_AP_anyhow(src)           # -> ['A','P']
                    # if "order_in_catalyst" in full_df.columns:
                        # df["order_in_cat"] = full_df["order_in_catalyst"]
            
                    df = df[["A", "P"]]
                    df = df.dropna(subset=["A", "P"])
                    if df.shape[1] not in [2,3]:
                        raise ValueError("DataFrame must have 2 or 3 columns after extraction.")
                        continue
                    max_A = df["A"].max()
                    if max_A is None or pd.isna(max_A) or max_A == 0:
                        raise ValueError("Max of A is invalid (None, NaN, or zero).")
                        continue
                    df["A"] = df["A"] / max_A
                    df["P"] = df["P"] / max_A   

                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
                    df.to_csv(dst, index=False, header=False)
                    print(f"Normalized and saved: {dst}")
                except Exception as e:
                    print(f"Error processing {src}: {e}")       
                continue

                if new_rect_height <= 0:
                    new_subs_rect = 0
                else:
                    new_subs_rect = (new_rect_height ** order) * time_diff

if __name__ == "__main__":
    start_iteration = 0
    end_iteration = 50
    mechanism_prefixes = ['M1_n', 'M1_cd', 'M1_scd', 'M1_pcd', 
                          'M1_c1d', 'M1_pc1d', 'M1_sc1d', 'M2_n', 
                          'M2_cd', 'M2_scd', 'M2_pcd', 'M2_c1d', 
                          'M2_pc1d', 'M2_sc1d', 'M2_c2d', 'M2_pc2d', 
                          'M2_sc2d', 'M3_n', 'M3_cd', 'M3_scd', 
                          'M3_pcd', 'M3_c1d', 'M3_pc1d', 'M3_sc1d', 
                          'M3_c2d', 'M3_pc2d', 'M3_sc2d']
                        
    normalize_and_save_to_training(start_iteration, end_iteration, mechanism_prefixes,
                                   source_root="/Users/dylanpyle/Coding/M1M2M3_data_Merged",
                                   dest_root="/Users/dylanpyle/Coding/RNN_Code/DATA/M1M2M3_data_Normalized",
                                   exps=(0,1))
    


