"""
Real Data Testing Module
========================

Standalone module for testing trained RNN models on real experimental data.
Imports model architectures and utilities from the main training module.
"""

import os
import numpy as np
from enum import Enum
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Import from your main module (adjust the import path as needed)
# Assuming your main module is named 'rnn_models' or similar
from RNN_Module import (
    EnhancedLSTM,
    EnhancedGRU,
)

# ==================== DATA LOADING UTILITIES ====================

def _read_AP_csv(csv_path: str) -> np.ndarray:
    """Fast CSV reader with minimal overhead for 2-column data."""
    df = pd.read_csv(csv_path, header=None, usecols=[0, 1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] != 2 or len(df) == 0:
        raise ValueError("CSV must have two numeric columns (A,P).")
    return df.values.astype(np.float32)

def read_duoconcat_csv(csv_path: str) -> np.ndarray:
    """Custom reader for duoconcat CSV files (4 columns, no headers)."""
    df = pd.read_csv(csv_path, header=None, usecols=[0, 1, 2, 3])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    
    if df.shape[1] != 4 or len(df) == 0:
        raise ValueError("DuoConcat CSV must have four numeric columns.")
    
    return df.values.astype(np.float32)

def load_csv_data(csv_path: str, expected_columns: int = 4) -> np.ndarray:
    """Load CSV data with automatic format detection."""
    try:
        if 'duoconcat' in csv_path.lower() or expected_columns == 4:
            return read_duoconcat_csv(csv_path)
        else:
            data = _read_AP_csv(csv_path)
            if expected_columns == 4 and data.shape[1] == 2:
                # Add derived features (gradients)
                A, P = data[:, 0], data[:, 1]
                dA_dt = np.gradient(A)
                dP_dt = np.gradient(P)
                data = np.column_stack([A, P, dA_dt, dP_dt])
            return data
    except Exception as e:
        raise ValueError(f"Error loading CSV data: {e}")

# ==================== DUOCONCAT CREATION ====================

def create_duoconcat_normalized(
    exp0_path: str, 
    exp1_path: str,
    output_path: str,
    mechanism_label: str,
    normalize_method: str = "minmax",
    concentration_cols: List[str] = None
) -> str:
    """
    Create a normalized duoconcat dataset from two experiment CSV files.
    
    Args:
        exp0_path: Path to first experiment CSV (Exp0.csv)
        exp1_path: Path to second experiment CSV (Exp1.csv) 
        output_path: Output path for duoconcat CSV
        mechanism_label: Mechanism label (e.g., "M1_cd")
        normalize_method: "minmax", "zscore", or "none"
        concentration_cols: Column names for concentrations [default: ["[SM]", "[P]"]]
        
    Returns:
        Path to created duoconcat file
    """
    
    if concentration_cols is None:
        concentration_cols = ["[SM]", "[P]"]
    
    print(f"=== CREATING DUOCONCAT DATASET ===")
    print(f"Exp0: {exp0_path}")
    print(f"Exp1: {exp1_path}")
    print(f"Mechanism: {mechanism_label}")
    print(f"Normalization: {normalize_method}")
    
    # Read both CSV files
    df0 = pd.read_csv(exp0_path)
    df1 = pd.read_csv(exp1_path)
    
    print(f"Exp0 shape: {df0.shape}")
    print(f"Exp1 shape: {df1.shape}")
    
    # Extract concentration data
    if not all(col in df0.columns for col in concentration_cols):
        missing = [col for col in concentration_cols if col not in df0.columns]
        raise ValueError(f"Missing columns in Exp0: {missing}")
    
    if not all(col in df1.columns for col in concentration_cols):
        missing = [col for col in concentration_cols if col not in df1.columns]
        raise ValueError(f"Missing columns in Exp1: {missing}")
    
    exp0_data = df0[concentration_cols].values.astype(np.float32)
    exp1_data = df1[concentration_cols].values.astype(np.float32)
    
    print(f"\n=== ORIGINAL RANGES ===")
    print(f"Exp0 [SM]: {np.min(exp0_data[:, 0]):.6f} to {np.max(exp0_data[:, 0]):.6f}")
    print(f"Exp0 [P]:  {np.min(exp0_data[:, 1]):.6f} to {np.max(exp0_data[:, 1]):.6f}")
    print(f"Exp1 [SM]: {np.min(exp1_data[:, 0]):.6f} to {np.max(exp1_data[:, 0]):.6f}")
    print(f"Exp1 [P]:  {np.min(exp1_data[:, 1]):.6f} to {np.max(exp1_data[:, 1]):.6f}")
    
    # Normalization function
    def normalize_data(data, method):
        if method == "minmax":
            data_min = np.min(data, axis=0, keepdims=True)
            data_max = np.max(data, axis=0, keepdims=True)
            data_range = data_max - data_min
            data_range = np.where(data_range == 0, 1, data_range)
            return (data - data_min) / data_range
        
        elif method == "zscore":
            data_mean = np.mean(data, axis=0, keepdims=True)
            data_std = np.std(data, axis=0, keepdims=True)
            data_std = np.where(data_std == 0, 1, data_std)
            return (data - data_mean) / data_std
        
        elif method == "none":
            return data
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    exp0_norm = normalize_data(exp0_data, normalize_method)
    exp1_norm = normalize_data(exp1_data, normalize_method)
    
    # Find minimum length
    min_length = min(len(exp0_norm), len(exp1_norm))
    print(f"\n=== DUOCONCAT ASSEMBLY ===")
    print(f"Using minimum length: {min_length}")
    
    exp0_trunc = exp0_norm[:min_length]
    exp1_trunc = exp1_norm[:min_length]
    
    # Create duoconcat: [A0, P0, A1, P1]
    duoconcat_data = np.concatenate([exp0_trunc, exp1_trunc], axis=1)
    
    print(f"Final duoconcat shape: {duoconcat_data.shape}")
    
    # Save without headers
    df_output = pd.DataFrame(duoconcat_data)
    df_output.to_csv(output_path, index=False, header=False)
    
    print(f"\n=== DUOCONCAT SAVED ===")
    print(f"File: {output_path}")
    
    return output_path

def auto_detect_device():
    """Auto-detect the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_model_instance(model_type: str, num_classes: int, **model_params) -> torch.nn.Module:
    """
    Create a model instance based on type string.
    
    Args:
        model_type: Type of model ("Enhanced_GRU", "Enhanced_LSTM", "Simplified_GRU", "Simplified_LSTM")
        num_classes: Number of output classes
        **model_params: Additional model parameters
    
    Returns:
        Model instance
    """
    # Default parameters
    default_params = {
        'input_size': 4,
        'hidden_size': 200,
        'num_layers': 3,
        'num_classes': num_classes,
        'dropout': 0.5,
    }
    
    # Enhanced model defaults
    enhanced_defaults = {
        'use_residual': True,
        'pooling_method': 'multi'
    }
    
    model_type_upper = model_type.upper()
    
    if model_type_upper in ["ENHANCED_GRU", "ENHANCEDGRU"]:
        params = {**default_params, **enhanced_defaults, **model_params}
        return EnhancedGRU(**params)
    
    elif model_type_upper in ["ENHANCED_LSTM", "ENHANCEDLSTM"]:
        params = {**default_params, **enhanced_defaults, **model_params}
        return EnhancedLSTM(**params)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported: Enhanced_GRU, Enhanced_LSTM, Simplified_GRU, Simplified_LSTM")

# ==================== MODEL TESTING FUNCTIONS ====================

# ==================== CONFIGURATION ====================
mech_list = ['a', 'ad', 'cd', 'n', 'pd']
# mech_list = ['M1_n', 'M2_n', 'M3_n', 'M1_cd', 'M2_cd', 'M3_cd']


class LabelMode(str, Enum):
    CE = "ce"

# ==================== UTILITY FUNCTIONS ====================

def extract_suffix(mechanism_name):
    """
    Extract the suffix from mechanism name (e.g., 'M1_sd' -> 'sd').
    
    Args:
        mechanism_name: Full mechanism name (e.g., 'M1_sd', 'M2_pd')
        
    Returns:
        Suffix string (e.g., 'sd', 'pd', 'n', 'a')
    """
    if '_' in mechanism_name:
        return mechanism_name.split('_')[1]
    else:
        return mechanism_name

def build_class_mapping(mech_list, use_suffix=True):
    """
    Build bidirectional class mapping.
    
    Args:
        mech_list: List of mechanism names
        use_suffix: If True, use only the suffix for classification
        
    Returns:
        class_to_idx: Dictionary mapping class names to indices
        idx_to_class: List mapping indices to class names
        full_to_suffix: Dictionary mapping full names to suffixes (if use_suffix=True)
    """
    if use_suffix:
        # Extract unique suffixes
        suffixes = sorted(set(extract_suffix(name) for name in mech_list))
        
        class_to_idx = {suffix: i for i, suffix in enumerate(suffixes)}
        idx_to_class = list(suffixes)
        
        # Create mapping from full name to suffix
        full_to_suffix = {name: extract_suffix(name) for name in mech_list}
        
        print(f"Using suffix-based labels: {idx_to_class}")
        print(f"Reduced from {len(mech_list)} classes to {len(suffixes)} classes")
        
        return class_to_idx, idx_to_class, full_to_suffix
    else:
        # Original behavior - use full names
        class_to_idx = {name: i for i, name in enumerate(mech_list)}
        idx_to_class = list(mech_list)
        
        return class_to_idx, idx_to_class, None

def get_label_from_mechanism(mechanism_name, class_to_idx, full_to_suffix=None):
    """
    Get the class index for a mechanism.
    
    Args:
        mechanism_name: Full mechanism name (e.g., 'M1_sd')
        class_to_idx: Dictionary mapping class names to indices
        full_to_suffix: Optional mapping from full names to suffixes
        
    Returns:
        Class index
    """
    if full_to_suffix is not None:
        # Use suffix
        suffix = full_to_suffix[mechanism_name]
        return class_to_idx[suffix]
    else:
        # Use full name
        return class_to_idx[mechanism_name]


def test_model(
    model_path: str,
    csv_path: str,
    expected_label: str,
    model_type: str = "Enhanced_LSTM",
    mech_list: List[str] = None,
    model_params: Dict = None,
    device: torch.device = None,
    return_logits: bool = False,
) -> Dict:
    """
    Load model and test on a single reaction file.
    
    Args:
        model_path: Path to trained model (.pth file)
        csv_path: Path to CSV data file
        expected_label: Expected mechanism label
        model_type: Type of model ("Enhanced_GRU", "Enhanced_LSTM", etc.)
        mech_list: List of mechanism labels
        model_params: Dictionary of model parameters to override defaults
        device: Device to use (auto-detected if None)
        return_logits: If True, return raw logits before softmax
    
    Returns:
        Dictionary with prediction results
    """
    
    # Set defaults
    if mech_list is None:
            mech_list = ['a', 'ad', 'cd', 'n', 'pd']    
    if device is None:
        device = auto_detect_device()
    
    if model_params is None:
        model_params = {}
    
    print(f"=== MODEL TESTING ON REAL DATA ===")
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"CSV: {csv_path}")
    print(f"Expected: {expected_label}")
    print(f"Return logits: {return_logits}")
    
    # Create model instance
    try:
        model = create_model_instance(
            model_type=model_type,
            num_classes=len(mech_list),
            **model_params
        )
        print(f"✓ Created {model_type} model")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return None
    
    # Load model weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✓ Loaded checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            state_dict = checkpoint
            print(f"✓ Loaded state dict")
        
        # Try strict loading first
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Model loaded (strict)")
        except Exception as strict_error:
            print(f"Strict loading failed, trying flexible loading...")
            
            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
            
            print(f"Loading {len(filtered_dict)}/{len(model_dict)} parameters")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)
            print(f"✓ Model loaded (partial)")
        
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Load data
    try:
        data = load_csv_data(csv_path, expected_columns=model.input_size)
        print(f"✓ Loaded data: {data.shape}")
        
        if data.shape[1] != model.input_size:
            raise ValueError(f"Data has {data.shape[1]} columns but model expects {model.input_size}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None
    
    # Convert to tensor
    try:
        x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        lengths = torch.tensor([len(data)], dtype=torch.long).to(device)
        
        print(f"Input shape: {x.shape}")
        print(f"Sequence length: {lengths.item()}")
        
    except Exception as e:
        print(f"✗ Error preparing tensors: {e}")
        return None
    
    # Make prediction
    try:
        with torch.no_grad():
            logits = model(x, lengths)
            logits_np = logits.cpu().numpy()[0]  # Raw logits
            
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_idx = int(torch.argmax(logits, dim=1).cpu().item())
            predicted_label = mech_list[predicted_idx]
            confidence = float(probabilities[predicted_idx])
        
        # Print results
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"Expected: {expected_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Correct: {'✓' if predicted_label == expected_label else '✗'}")
        
        if return_logits:
            print(f"\nRaw logits:")
            for i, (mech, logit) in enumerate(zip(mech_list, logits_np)):
                marker = " ← PREDICTED" if i == predicted_idx else ""
                print(f"  {mech}: {logit:.6f}{marker}")
        
        print(f"\nProbabilities:")
        for i, (mech, prob) in enumerate(zip(mech_list, probabilities)):
            marker = " ← PREDICTED" if i == predicted_idx else ""
            print(f"  {mech}: {prob:.3f}{marker}")
        
        result = {
            "predicted": predicted_label,
            "expected": expected_label,
            "confidence": confidence,
            "correct": predicted_label == expected_label,
            "probabilities": dict(zip(mech_list, probabilities)),
            "model_type": model_type,
            "data_shape": data.shape,
            "sequence_length": len(data)
        }
        
        if return_logits:
            result["logits"] = dict(zip(mech_list, logits_np))
            result["raw_logits"] = logits_np
        
        return result
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        return None

def test_duoconcat_with_model(
    duoconcat_path: str,
    model_path: str,
    expected_label: str,
    model_type: str = "Enhanced_GRU",
    return_logits: bool = False,
    **kwargs
) -> Dict:
    """
    Test duoconcat file with model.
    
    Args:
        duoconcat_path: Path to duoconcat CSV
        model_path: Path to trained model
        expected_label: Expected mechanism label
        model_type: Type of model
        return_logits: If True, return raw logits
        **kwargs: Additional arguments for test_model
    
    Returns:
        Dictionary with test results
    """
    print(f"\n=== TESTING DUOCONCAT WITH MODEL ===")
    print(f"File: {duoconcat_path}")
    print(f"Expected: {expected_label}")
    
    return test_model(
        model_path=model_path,
        csv_path=duoconcat_path,
        expected_label=expected_label,
        model_type=model_type,
        return_logits=return_logits,
        **kwargs
    )

def test_multiple_files(
    model_path: str,
    test_files: List[Tuple[str, str]],
    model_type: str = "Enhanced_GRU",
    return_logits: bool = False,
    **kwargs
) -> Dict:
    """
    Test model on multiple files.
    
    Args:
        model_path: Path to trained model
        test_files: List of (csv_path, expected_label) tuples
        model_type: Type of model
        return_logits: If True, include raw logits in results
        **kwargs: Additional arguments for test_model
    
    Returns:
        Dictionary with summary results
    """
    
    print(f"=== BATCH TESTING: {len(test_files)} FILES ===")
    
    results = []
    correct_count = 0
    
    for i, (csv_path, expected_label) in enumerate(test_files, 1):
        print(f"\n--- Test {i}/{len(test_files)} ---")
        
        result = test_model(
            model_path=model_path,
            csv_path=csv_path,
            expected_label=expected_label,
            model_type=model_type,
            return_logits=return_logits,
            **kwargs
        )
        
        if result is not None:
            results.append(result)
            if result['correct']:
                correct_count += 1
        else:
            results.append({
                'csv_path': csv_path,
                'expected': expected_label,
                'predicted': 'ERROR',
                'correct': False
            })
    
    # Summary
    total_tests = len(results)
    accuracy = correct_count / total_tests if total_tests > 0 else 0
    avg_confidence = np.mean([r.get('confidence', 0) for r in results])
    
    summary = {
        'total_tests': total_tests,
        'correct_predictions': correct_count,
        'accuracy': accuracy,
        'average_confidence': avg_confidence,
        'individual_results': results
    }
    
    print(f"\n=== BATCH TESTING SUMMARY ===")
    print(f"Total: {total_tests}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Avg confidence: {avg_confidence:.3f}")
    
    return summary

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example 1: Create duoconcat from real data
    duoconcat_file = create_duoconcat_normalized(
        exp0_path="/Users/dylanpyle/VsCode/RNN_Code/DATA/EVAL_REAL_DATA/Wes_Data/Os_High(Sheet1)(1).csv", 
        exp1_path='/Users/dylanpyle/VsCode/RNN_Code/DATA/EVAL_REAL_DATA/Wes_Data/Os_Low(Sheet1)(1).csv',
        output_path='/Users/dylanpyle/VsCode/M1_pd_duoconcat.csv',
        mechanism_label='pd',
        normalize_method="minmax"
    )
    
    # Example 2: Test with model and get raw logits
    result = test_model(
        model_path="/Users/dylanpyle/VsCode/checkpoints/DeactAI_full_200x3_best.pth",
        csv_path="/Users/dylanpyle/VsCode/M1_pd_duoconcat.csv",
        expected_label="pd",
        model_type="Enhanced_LSTM",
        model_params={
            'hidden_size': 200,
            'num_layers': 3,
            'dropout': 0.3,
            'use_residual': True,
            'pooling_method': 'multi'
        },
        return_logits=True  # Get raw logits before softmax
    )
    
    # Access raw logits
    if result and 'raw_logits' in result:
        print("\n=== RAW LOGITS ARRAY ===")
        print(result['raw_logits'])