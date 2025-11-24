import os
import time
import math
import gc
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ==================== CONFIGURATION ====================
mech_list = ['M1a_n', 'M1a_cd','M1_n', 'M1_cd', 'M1a_sd', 'M1_sd', 'M1a_pd', 'M1_pd']
            # 'M2a_n', 'M2a_cd', 'M2a_sd', 'M2a_pd', 'M2_n', 'M2_cd', 'M2_sd', 'M2_pd']
# mech_list = ['M1_n', 'M1_cd', 'M1_sd', 'M1_pd']


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

def _read_AP_csv(csv_path: str) -> np.ndarray:
    """Fast CSV reader with minimal overhead."""
    df = pd.read_csv(csv_path, header=None, usecols=[0, 1])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] != 2 or len(df) == 0:
        raise ValueError("CSV must have two numeric columns (A,P).")
    return df.values.astype(np.float32)

def get_data_filtered_duos(
    mech_list,
    start_iteration,
    end_iteration,
    exps=(0, 1),
    min_time_points=8,
    data_base_dir="/Users/dylanpyle/Coding/RNN_Code/DATA/PT_activation",
    json_base_dir="/Users/dylanpyle/Coding/M1M2_data_merged",
    verbose=True,
):
    """Optimized data loader with better error handling."""
    X_list, y_name_list, P_list = [], [], []
    reasons = {"missing": 0, "parse": 0, "short": 0, "low_yield": 0}
    total_found, total_skipped = 0, 0
    
    for i in range(start_iteration, end_iteration + 1):
        for name in mech_list:
            base = os.path.join(data_base_dir, name, f"rct_{i}")
            csv0 = os.path.join(base, f"exp_{exps[0]}.csv")
            csv1 = os.path.join(base, f"exp_{exps[1]}.csv")
            json_path = os.path.join(json_base_dir, name, f"rct_{i}", "info_dict.json")
            
            if not (os.path.exists(csv0) and os.path.exists(csv1)):
                reasons["missing"] += 1
                total_skipped += 1
                continue
            
            try:
                d0 = _read_AP_csv(csv0)
                d1 = _read_AP_csv(csv1)
                
                # Check if final value in second column (index 1) is < 0.5 for both experiments
                if d0[-1, 1] < 0.5 or d1[-1, 1] < 0.5:
                    reasons["low_yield"] += 1
                    total_skipped += 1
                    continue
                
                if len(d0) < min_time_points or len(d1) < min_time_points:
                    reasons["short"] += 1
                    total_skipped += 1
                    continue
                
                T = min(len(d0), len(d1))
                x = np.concatenate([d0[:T], d1[:T]], axis=1)
                X_list.append(x)
                y_name_list.append(name)
                P_list.append(json_path)
                total_found += 1
                
            except Exception:
                reasons["parse"] += 1
                total_skipped += 1
    
    if verbose:
        print(f"\n[DUO] Found {total_found} valid samples. Skipped {total_skipped}.")
        if total_skipped:
            print("Skip reasons: " + " | ".join(f"{k}={reasons[k]}" for k in reasons))
    
    if not X_list:
        raise ValueError("No valid DUO datasets found.")
    
    return X_list, y_name_list, P_list

# ==================== DYNAMIC RESAMPLING PROTOCOLS ====================
class ResamplingStrategy(str, Enum):
    """Different resampling strategies for temporal data."""
    UNIFORM = "uniform"          # Uniform sampling between first and last
    EXPONENTIAL = "exponential"  # More points early, fewer later
    ADAPTIVE = "adaptive"        # Based on rate of change
    RANDOM = "random"           # Random selection with constraints

def resample_sequence(
    sequence: np.ndarray,
    target_length: int,
    strategy: ResamplingStrategy = ResamplingStrategy.UNIFORM,
    preserve_endpoints: bool = True,
    min_points: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a temporal sequence to a target length while preserving temporal structure.
    
    Args:
        sequence: (T, features) temporal sequence
        target_length: desired output length
        strategy: resampling strategy
        preserve_endpoints: always keep first and last points
        min_points: minimum number of points to sample
    
    Returns:
        resampled_sequence: (target_length, features)
        original_indices: indices from original sequence
    """
    original_length = len(sequence)
    
    # Handle edge cases
    if target_length >= original_length:
        # If target is longer or equal, return original with padding if needed
        if target_length == original_length:
            return sequence.copy(), np.arange(original_length)
        else:
            # Interpolate to increase length
            return interpolate_sequence(sequence, target_length), np.linspace(0, original_length-1, target_length)
    
    if target_length < min_points:
        target_length = min_points
    
    if preserve_endpoints and target_length >= 2:
        # Always include first and last points
        if target_length == 2:
            indices = np.array([0, original_length - 1])
        else:
            # Sample target_length-2 points from the middle
            middle_points = target_length - 2
            
            if strategy == ResamplingStrategy.UNIFORM:
                # Uniform sampling of middle points
                if middle_points > 0:
                    middle_indices = np.linspace(1, original_length - 2, middle_points, dtype=int)
                    indices = np.concatenate([[0], middle_indices, [original_length - 1]])
                else:
                    indices = np.array([0, original_length - 1])
                    
            elif strategy == ResamplingStrategy.EXPONENTIAL:
                # More points early in the sequence
                if middle_points > 0:
                    # Exponential spacing (more points early)
                    exp_points = np.exp(np.linspace(0, 1, middle_points)) - 1
                    exp_points = exp_points / exp_points.max()  # Normalize to [0, 1]
                    middle_indices = (exp_points * (original_length - 3) + 1).astype(int)
                    middle_indices = np.unique(middle_indices)  # Remove duplicates
                    
                    # If we lost points due to rounding, add more uniformly
                    while len(middle_indices) < middle_points and len(middle_indices) < original_length - 2:
                        remaining = middle_points - len(middle_indices)
                        available_indices = set(range(1, original_length - 1)) - set(middle_indices)
                        if available_indices:
                            additional = np.random.choice(list(available_indices), 
                                                        min(remaining, len(available_indices)), 
                                                        replace=False)
                            middle_indices = np.concatenate([middle_indices, additional])
                    
                    middle_indices = np.sort(middle_indices)
                    indices = np.concatenate([[0], middle_indices, [original_length - 1]])
                else:
                    indices = np.array([0, original_length - 1])
                    
            elif strategy == ResamplingStrategy.ADAPTIVE:
                # Sample based on rate of change
                if middle_points > 0:
                    # Calculate rate of change (simple difference between consecutive points)
                    if sequence.ndim == 1:
                        changes = np.abs(np.diff(sequence))
                    else:
                        changes = np.linalg.norm(np.diff(sequence, axis=0), axis=1)
                    
                    # Normalize changes to probabilities
                    if changes.sum() > 0:
                        probs = changes / changes.sum()
                        
                        # Sample indices based on change probability
                        middle_indices = np.random.choice(
                            range(1, original_length - 1), 
                            size=min(middle_points, original_length - 2),
                            replace=False,
                            p=probs[:-1] / probs[:-1].sum()  # Exclude last change (it's at index original_length-1)
                        )
                    else:
                        # If no changes, fall back to uniform
                        middle_indices = np.linspace(1, original_length - 2, middle_points, dtype=int)
                    
                    middle_indices = np.sort(middle_indices)
                    indices = np.concatenate([[0], middle_indices, [original_length - 1]])
                else:
                    indices = np.array([0, original_length - 1])
                    
            elif strategy == ResamplingStrategy.RANDOM:
                # Random sampling of middle points
                if middle_points > 0:
                    available_middle = list(range(1, original_length - 1))
                    if len(available_middle) >= middle_points:
                        middle_indices = np.sort(np.random.choice(
                            available_middle, size=middle_points, replace=False
                        ))
                    else:
                        middle_indices = np.array(available_middle)
                    
                    indices = np.concatenate([[0], middle_indices, [original_length - 1]])
                else:
                    indices = np.array([0, original_length - 1])
    else:
        # Don't preserve endpoints
        indices = np.linspace(0, original_length - 1, target_length, dtype=int)
    
    # Remove duplicates and sort
    indices = np.unique(indices)
    
    # Ensure we don't exceed target length
    if len(indices) > target_length:
        # Keep first, last, and sample from middle
        if preserve_endpoints:
            middle_indices = indices[1:-1]
            if len(middle_indices) > target_length - 2:
                keep_middle = np.random.choice(middle_indices, target_length - 2, replace=False)
                indices = np.concatenate([[indices[0]], np.sort(keep_middle), [indices[-1]]])
        else:
            indices = indices[:target_length]
    
    resampled_sequence = sequence[indices]
    return resampled_sequence, indices

def interpolate_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """Interpolate sequence to increase its length."""
    original_length = len(sequence)
    
    if sequence.ndim == 1:
        # 1D sequence
        original_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, target_length)
        return np.interp(new_indices, original_indices, sequence)
    else:
        # Multi-dimensional sequence
        original_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, target_length)
        
        interpolated = np.zeros((target_length, sequence.shape[1]))
        for feature_idx in range(sequence.shape[1]):
            interpolated[:, feature_idx] = np.interp(
                new_indices, original_indices, sequence[:, feature_idx]
            )
        return interpolated

def dynamic_collate_fn(batch):
    """Collate function for dynamic length sequences."""
    sequences, labels, lengths = zip(*batch)
    
    # Find maximum length in this batch
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to max length in batch
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.zeros((max_len - len(seq), seq.shape[1]), dtype=torch.float32)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack everything
    x_batch = torch.stack(padded_sequences)
    length_batch = torch.stack(lengths)
    y_batch = torch.stack(labels) if labels[0].ndim > 0 else torch.tensor(labels, dtype=torch.long)
    
    return x_batch, y_batch, length_batch

# ==================== SIMPLIFIED COLLATE FUNCTION ====================
def simple_collate_fn(batch):
    """Simple collate function for fixed-size data."""
    xs, ys, lengths = zip(*batch)
    
    # Stack tensors directly since they're already the same size
    x_batch = torch.stack(xs)
    length_batch = torch.stack(lengths)
    y_batch = torch.stack(ys) if ys[0].ndim > 0 else torch.tensor(ys, dtype=torch.long)
    
    return x_batch, y_batch, length_batch

# ==================== ENHANCED POOLING MODULES ====================

class AttentionPooling(nn.Module):
    """Learnable attention pooling for sequence aggregation."""
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, hidden_states, lengths):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            lengths: (batch,) actual lengths of sequences
        """
        batch_size, max_len, hidden_size = hidden_states.shape
        
        # Create attention weights
        attention_weights = self.attention(hidden_states)  # (batch, seq_len, 1)
        attention_weights = attention_weights.squeeze(-1)  # (batch, seq_len)
        
        # Create mask for valid positions
        mask = torch.arange(max_len, device=hidden_states.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        
        # Apply mask (set invalid positions to very negative value)
        attention_weights = attention_weights.masked_fill(~mask, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_weights, dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        pooled = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), dim=1
        )  # (batch, hidden_size)
        
        return pooled

class MultiPooling(nn.Module):
    """Multiple pooling strategies combined."""
    
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention_pooling = AttentionPooling(hidden_size, dropout)
        self.projection = nn.Linear(hidden_size * 3, hidden_size)  # 3 pooling methods
        
    def forward(self, hidden_states, lengths):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            lengths: (batch,) actual lengths of sequences
        """
        batch_size, max_len, hidden_size = hidden_states.shape
        
        # Create mask for valid positions
        mask = torch.arange(max_len, device=hidden_states.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)
        
        # 1. Attention pooling
        attention_pooled = self.attention_pooling(hidden_states, lengths)
        
        # 2. Mean pooling (over valid positions)
        masked_states = hidden_states * mask.unsqueeze(-1).float()
        mean_pooled = masked_states.sum(dim=1) / lengths.unsqueeze(-1).float()
        
        # 3. Last valid position
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        last_indices = (lengths - 1).clamp(min=0)
        last_pooled = hidden_states[batch_indices, last_indices]
        
        # Combine all pooling methods
        combined = torch.cat([attention_pooled, mean_pooled, last_pooled], dim=1)
        
        # Project back to original size
        pooled = self.projection(combined)
        
        return pooled

# =========== DATASETS ==================
class DuoDataset(Dataset):
    """Fixed dataset without variable length augmentation - preserves original data."""
    
    def __init__(
        self, 
        X_list, 
        y_name_list, 
        mech_list, 
        label_mode: LabelMode = LabelMode.CE, 
        max_length=None,
        use_suffix: bool = True  # NEW: Use suffix-based labels
    ):
        self.mech_list = mech_list
        self.label_mode = LabelMode(label_mode)
        self.use_suffix = use_suffix
        
        # Build class mapping with suffix support
        self.class_to_idx, self.idx_to_class, self.full_to_suffix = build_class_mapping(
            mech_list, use_suffix=use_suffix
        )
        
        # Convert to tensors with consistent length
        if max_length is None:
            max_length = max(len(x) for x in X_list)
        self.max_length = max_length
        
        self.X = []
        self.lengths = []
        self.label_names = []  # Store original label names
        
        for i, x in enumerate(X_list):
            x_tensor = torch.as_tensor(x, dtype=torch.float32)
            actual_length = min(len(x), max_length)
            
            # Pad or truncate to max_length
            if len(x) < max_length:
                padding = torch.zeros((max_length - len(x), x.shape[1]), dtype=torch.float32)
                x_padded = torch.cat([x_tensor, padding], dim=0)
            else:
                x_padded = x_tensor[:max_length]
            
            self.X.append(x_padded)
            self.lengths.append(actual_length)
            self.label_names.append(y_name_list[i])
        
        self.X = torch.stack(self.X)
        self.lengths = torch.tensor(self.lengths, dtype=torch.long)
        
        # Convert labels - NOW USING SUFFIX MAPPING
        if self.label_mode == LabelMode.CE:
            if use_suffix and self.full_to_suffix is not None:
                # Convert full mechanism names to suffix-based labels
                self.Y = torch.tensor([
                    self.class_to_idx[self.full_to_suffix[name]] 
                    for name in self.label_names
                ], dtype=torch.long)
            else:
                # Use full mechanism names
                self.Y = torch.tensor([
                    self.class_to_idx[name] 
                    for name in self.label_names
                ], dtype=torch.long)
        else:
            K = len(self.idx_to_class)  # Use actual number of classes (suffixes)
            Y = torch.zeros((len(self.X), K), dtype=torch.float32)
            for i, name in enumerate(self.label_names):
                if use_suffix and self.full_to_suffix is not None:
                    suffix = self.full_to_suffix[name]
                    Y[i, self.class_to_idx[suffix]] = 1.0
                else:
                    Y[i, self.class_to_idx[name]] = 1.0
            self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.lengths[i]
    
    def get_label_name(self, idx):
        """Get the label name (suffix or full name) for a given index."""
        full_name = self.label_names[idx]
        if self.use_suffix and self.full_to_suffix is not None:
            return self.full_to_suffix[full_name]
        else:
            return full_name


class DynamicDuoDataset(Dataset):
    """Dataset with dynamic sequence length through resampling."""
    
    def __init__(
        self, 
        X_list, 
        y_name_list, 
        mech_list, 
        label_mode: LabelMode = LabelMode.CE,
        min_length: int = 5,
        max_length: int = 75, 
        resampling_strategy: ResamplingStrategy = ResamplingStrategy.UNIFORM,
        resample_probability: float = 0.8,
        preserve_endpoints: bool = True,
        length_distribution: str = "uniform",
        use_suffix: bool = True  
    ):
        self.mech_list = mech_list
        self.label_mode = LabelMode(label_mode)
        self.use_suffix = use_suffix
        
        # Build class mapping with suffix 
        self.class_to_idx, self.idx_to_class, self.full_to_suffix = build_class_mapping(
            mech_list, use_suffix=use_suffix
        )
        
        self.min_length = min_length
        self.max_length = max_length
        self.resampling_strategy = ResamplingStrategy(resampling_strategy)
        self.resample_probability = resample_probability
        self.preserve_endpoints = preserve_endpoints
        self.length_distribution = length_distribution

        # Store original sequences
        self.original_X = []
        self.original_lengths = []
        self.label_names = []  # Store original label names
        
        for i, x in enumerate(X_list):
            x_tensor = torch.as_tensor(x, dtype=torch.float32)
            # Only keep sequences that are long enough to be meaningfully resampled
            if len(x) >= min_length:
                self.original_X.append(x_tensor)
                self.original_lengths.append(len(x))
                self.label_names.append(y_name_list[i])
        
        print(f"Filtered to {len(self.original_X)} sequences with length >= {min_length}")
        print(f"Original length range: {min(self.original_lengths)} to {max(self.original_lengths)}")
        
        # Convert labels - NOW USING SUFFIX MAPPING
        if self.label_mode == LabelMode.CE:
            if use_suffix and self.full_to_suffix is not None:
                # Convert full mechanism names to suffix-based labels
                self.Y = torch.tensor([
                    self.class_to_idx[self.full_to_suffix[name]] 
                    for name in self.label_names
                ], dtype=torch.long)
            else:
                # Use full mechanism names
                self.Y = torch.tensor([
                    self.class_to_idx[name] 
                    for name in self.label_names
                ], dtype=torch.long)
        else:
            K = len(self.idx_to_class)  # Use actual number of classes (suffixes)
            Y = torch.zeros((len(self.original_X), K), dtype=torch.float32)
            for i, name in enumerate(self.label_names):
                if use_suffix and self.full_to_suffix is not None:
                    suffix = self.full_to_suffix[name]
                    Y[i, self.class_to_idx[suffix]] = 1.0
                else:
                    Y[i, self.class_to_idx[name]] = 1.0
            self.Y = Y

    def _sample_target_length(self):
        """Sample target length according to specified distribution."""
        if self.length_distribution == "uniform":
            return np.random.randint(self.min_length, self.max_length + 1)
        elif self.length_distribution == "normal":
            mean = (self.min_length + self.max_length) / 2
            std = (self.max_length - self.min_length) / 6
            length = int(np.random.normal(mean, std))
            return np.clip(length, self.min_length, self.max_length)
        elif self.length_distribution == "exponential":
            scale = (self.max_length - self.min_length) / 3
            length = int(np.random.exponential(scale) + self.min_length)
            return np.clip(length, self.min_length, self.max_length)
        else:
            return np.random.randint(self.min_length, self.max_length + 1)

    def __len__(self):
        return len(self.original_X)

    def __getitem__(self, idx):
        # Get original sequence
        original_seq = self.original_X[idx].numpy()
        original_length = self.original_lengths[idx]
        
        # Decide whether to resample
        if np.random.random() < self.resample_probability and original_length > self.min_length:
            target_length = self._sample_target_length()
            
            if target_length >= original_length:
                seq_tensor = self.original_X[idx]
                actual_length = original_length
            else:
                resampled_seq, _ = resample_sequence(
                    original_seq,
                    target_length,
                    strategy=self.resampling_strategy,
                    preserve_endpoints=self.preserve_endpoints
                )
                seq_tensor = torch.as_tensor(resampled_seq, dtype=torch.float32)
                actual_length = len(resampled_seq)
        else:
            if original_length > self.max_length:
                if self.preserve_endpoints:
                    truncated_seq, _ = resample_sequence(
                        original_seq,
                        self.max_length,
                        strategy=self.resampling_strategy,
                        preserve_endpoints=True
                    )
                    seq_tensor = torch.as_tensor(truncated_seq, dtype=torch.float32)
                    actual_length = len(truncated_seq)
                else:
                    seq_tensor = self.original_X[idx][:self.max_length]
                    actual_length = self.max_length
            else:
                seq_tensor = self.original_X[idx]
                actual_length = original_length
        
        return seq_tensor, self.Y[idx], torch.tensor(actual_length, dtype=torch.long)
    
    def get_label_name(self, idx):
        """Get the label name (suffix or full name) for a given index."""
        full_name = self.label_names[idx]
        if self.use_suffix and self.full_to_suffix is not None:
            return self.full_to_suffix[full_name]
        else:
            return full_name

# ==================== RESIDUAL LAYERS ====================
class ResidualGRULayer(nn.Module):
    """GRU layer with residual connections."""
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Projection layer if input and hidden sizes don't match
        self.projection = None
        if input_size != hidden_size:
            self.projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        gru_out, hidden = self.gru(x)
        gru_out = self.dropout(gru_out)
        
        # Residual connection
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x
            
        # Add residual and apply layer norm
        output = self.layer_norm(gru_out + residual)
        
        return output, hidden

class ResidualLSTMLayer(nn.Module):
    """LSTM layer with residual connections."""
    
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Projection layer if input and hidden sizes don't match
        self.projection = None
        if input_size != hidden_size:
            self.projection = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Residual connection
        if self.projection is not None:
            residual = self.projection(x)
        else:
            residual = x
            
        # Add residual and apply layer norm
        output = self.layer_norm(lstm_out + residual)
        
        return output, (hidden, cell)

# ==================== ENHANCED MODELS WITH POOLING AND RESIDUALS ====================
class EnhancedGRU(nn.Module):
    """Enhanced GRU with residual connections and advanced pooling."""
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_residual: bool = True,
        pooling_method: str = "multi"  # "attention", "multi", "last"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.pooling_method = pooling_method
        
        # Build layers
        if use_residual:
            self.layers = nn.ModuleList()
            
            # First layer (input_size -> hidden_size)
            self.layers.append(ResidualGRULayer(input_size, hidden_size, dropout))
            
            # Subsequent layers (hidden_size -> hidden_size)
            for _ in range(num_layers - 1):
                self.layers.append(ResidualGRULayer(hidden_size, hidden_size, dropout))
        else:
            # Standard GRU
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Pooling layer
        if pooling_method == "attention":
            self.pooling = AttentionPooling(hidden_size, dropout)
        elif pooling_method == "multi":
            self.pooling = MultiPooling(hidden_size, dropout)
        else:  # "last"
            self.pooling = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        if hasattr(self, 'gru'):
            for name, param in self.gru.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    n = param.size(0)
                    param.data[n//3:2*n//3].fill_(1.)  # Reset gate bias
        else:
            # Residual layers case
            for layer in self.layers:
                if hasattr(layer, 'gru'):
                    for name, param in layer.gru.named_parameters():  # type: ignore
                        if 'weight' in name:
                            nn.init.orthogonal_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
                            n = param.size(0)
                            param.data[n//3:2*n//3].fill_(1.)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths):
        """Forward pass with enhanced architecture."""
        batch_size = x.size(0)
        
        # Process through layers
        if self.use_residual:
            hidden_states = x
            for layer in self.layers:
                hidden_states, _ = layer(hidden_states)
        else:
            hidden_states, _ = self.gru(x)
        
        # Apply pooling
        if self.pooling is not None:
            pooled = self.pooling(hidden_states, lengths)
        else:
            # Use last valid output
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = (lengths - 1).clamp(min=0)
            pooled = hidden_states[batch_indices, last_indices]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

# ==================== GRADIENT CHECKING UTILITIES ====================

def check_gradients(model, train_loader, device):
    """Check if gradients are flowing properly."""
    # Ensure model is on the correct device
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Get one batch
    x, y, lengths = next(iter(train_loader))
    x, y, lengths = x.to(device), y.to(device), lengths.to(device)
    
    # Clear any existing gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(x, lengths)
    loss = criterion(outputs, y)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            grad_info[name] = {
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'ratio': grad_norm / (param_norm + 1e-8)
            }
        else:
            grad_info[name] = {'grad_norm': 0, 'param_norm': param.norm().item(), 'ratio': 0}
    
    return grad_info

def print_gradient_info(grad_info):
    """Print gradient information in a readable format."""
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    print(f"{'Layer Name':<30} {'Grad Norm':<12} {'Param Norm':<12} {'Ratio':<12}")
    print("-"*80)
    
    for name, info in grad_info.items():
        print(f"{name:<30} {info['grad_norm']:<12.2e} {info['param_norm']:<12.2e} {info['ratio']:<12.2e}")
    
    # Check for potential issues
    print("\n" + "="*50)
    print("POTENTIAL ISSUES:")
    print("="*50)
    
    zero_grad_layers = [name for name, info in grad_info.items() if info['grad_norm'] < 1e-10]
    if zero_grad_layers:
        print(f"⚠️  Layers with zero gradients: {zero_grad_layers}")
    
    large_grad_layers = [name for name, info in grad_info.items() if info['grad_norm'] > 10]
    if large_grad_layers:
        print(f"⚠️  Layers with large gradients: {large_grad_layers}")
    
    small_ratio_layers = [name for name, info in grad_info.items() if info['ratio'] < 1e-6]
    if small_ratio_layers:
        print(f"⚠️  Layers with very small gradient/parameter ratios: {small_ratio_layers}")
    
    if not zero_grad_layers and not large_grad_layers and not small_ratio_layers:
        print("✅ Gradient flow looks healthy!")

# ==================== EXAMPLE USAGE ====================

def Train_Enhanced_Models():
    
    # Load data (same as before)
    print("Loading data...")
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=0,
        end_iteration=35000,
        min_time_points=5
    )
    
    # Create dataset
    dataset = DuoDataset(
        X_list, 
        y_name_list, 
        mech_list,
        label_mode=LabelMode.CE,
        max_length=200,
        use_suffix=True  # Default: combines mechanisms with same suffix
        )

    print(f"Dataset size: {len(dataset)}")
    print(f"Input shape: {dataset.X.shape}")  # type: ignore
    print(f"Max sequence length: {dataset.max_length}")
    
    # Split data
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, collate_fn=simple_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=simple_collate_fn)
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create enhanced model
    model = EnhancedLSTM(
        input_size=4,  
        hidden_size=256,
        num_layers=3,
        num_classes=5,
        dropout=0.3,
        use_residual=True,
        pooling_method="multi"  # Try "attention", "multi", or "last"
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Move model to device before gradient checking
    model = model.to(device)
    
    # Check gradients before training
    print("\nChecking gradient flow...")
    grad_info = check_gradients(model, train_loader, device)
    print_gradient_info(grad_info)
    
    # Train model
    history = train_simple_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1500,
        learning_rate=1e-4,
        device=device,
        model_name="Deact_AI_Full_Package_3x256"
    )
    
    return model, history

# ==================== MODEL COMPARISON FUNCTION ====================
def analyze_length_distributions():
    """Analyze how different length distributions affect the dataset."""
    
    # Load small dataset for testing
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=0,
        end_iteration=3000,
        min_time_points=8
    )
    
    distributions = ["uniform", "normal", "exponential"]
    
    print("Analyzing length distributions for 8-40 point range...")
    print("="*60)
    
    for dist in distributions:
        print(f"\n{dist.upper()} Distribution:")
        
        dataset = DynamicDuoDataset(
            X_list, y_name_list, mech_list,
            min_length=8, max_length=40,
            resampling_strategy=ResamplingStrategy.UNIFORM,
            resample_probability=1.0,  # Always resample for analysis
            length_distribution=dist
        )
        
        # Sample many lengths from first sequence
        lengths = []
        for _ in range(1000):
            _, _, length = dataset[0]
            lengths.append(length.item())
        
        lengths = np.array(lengths)
        print(f"  Mean: {lengths.mean():.1f}")
        print(f"  Std:  {lengths.std():.1f}")
        print(f"  Min:  {lengths.min()}")
        print(f"  Max:  {lengths.max()}")
        
        # Show distribution
        bins = np.arange(7.5, 41.5, 1)  # Bins for 8-40
        hist, _ = np.histogram(lengths, bins=bins)
        print(f"  Distribution: {dict(zip(range(8, 41), hist))}")

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with residual connections and advanced pooling."""
    
    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_residual: bool = True,
        pooling_method: str = "multi"  # "attention", "multi", "last"
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.pooling_method = pooling_method
        
        # Build layers
        if use_residual:
            self.layers = nn.ModuleList()
            
            # First layer (input_size -> hidden_size)
            self.layers.append(ResidualLSTMLayer(input_size, hidden_size, dropout))
            
            # Subsequent layers (hidden_size -> hidden_size)
            for _ in range(num_layers - 1):
                self.layers.append(ResidualLSTMLayer(hidden_size, hidden_size, dropout))
        else:
            # Standard LSTM
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # Pooling layer
        if pooling_method == "attention":
            self.pooling = AttentionPooling(hidden_size, dropout)
        elif pooling_method == "multi":
            self.pooling = MultiPooling(hidden_size, dropout)
        else:  # "last"
            self.pooling = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        if hasattr(self, 'lstm'):
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.)  # Forget gate bias
        else:
            # Residual layers case
            for layer in self.layers:
                if hasattr(layer, 'lstm'):
                    for name, param in layer.lstm.named_parameters():  # type: ignore
                        if 'weight' in name:
                            nn.init.orthogonal_(param)
                        elif 'bias' in name:
                            nn.init.constant_(param, 0)
                            n = param.size(0)
                            param.data[n//4:n//2].fill_(1.)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths):
        """Forward pass with enhanced architecture."""
        batch_size = x.size(0)
        
        # Process through layers
        if self.use_residual:
            hidden_states = x
            for layer in self.layers:
                hidden_states, _ = layer(hidden_states)
        else:
            hidden_states, _ = self.lstm(x)
        
        # Apply pooling
        if self.pooling is not None:
            pooled = self.pooling(hidden_states, lengths)
        else:
            # Use last valid output
            batch_indices = torch.arange(batch_size, device=x.device)
            last_indices = (lengths - 1).clamp(min=0)
            pooled = hidden_states[batch_indices, last_indices]
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits

def train_simple_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-3,
    device: Union[str, torch.device] = 'cpu',
    save_dir='./checkpoints',
    model_name='simple_model',
    scheduler_type='plateau'  # 'plateau', 'cosine', 'step', 'onecycle'
):
    """Simplified training function with better gradient tracking and flexible schedulers."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Create scheduler based on type
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        needs_metric = True
        step_per_batch = False
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        needs_metric = False
        step_per_batch = False
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.5
        )
        needs_metric = False
        step_per_batch = False
    elif scheduler_type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate * 10,
            steps_per_epoch=len(train_loader), 
            epochs=num_epochs,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        needs_metric = False
        step_per_batch = True
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")
    
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [],
        'learning_rates': []
    }
    
    def compute_accuracy(loader):
        model.eval()
        correct, total = 0, 0
        total_loss = 0
        
        with torch.no_grad():
            for x, y, lengths in loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                
                outputs = model(x, lengths)
                loss = criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return total_loss / len(loader), correct / total
    
    print(f"Starting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Scheduler: {scheduler_type}")
    print(f"Initial LR: {learning_rate:.2e}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (x, y, lengths) in enumerate(pbar):
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step scheduler if it needs to step per batch (OneCycleLR)
            if step_per_batch:
                scheduler.step()  # type: ignore
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            # Update progress bar
            current_acc = correct / total
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{train_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        # Calculate epoch metrics
        train_loss /= len(train_loader)
        train_acc = correct / total
        
        # Validation
        if val_loader:
            val_loss, val_acc = compute_accuracy(val_loader)
        else:
            val_loss, val_acc = 0, 0
        
        # Step scheduler if it needs to step per epoch
        if not step_per_batch:
            if needs_metric:
                # ReduceLROnPlateau needs a metric - convert to float if needed
                metric = val_loss if val_loader else train_loss
                if isinstance(metric, torch.Tensor):
                    metric = metric.item()
                scheduler.step(metric)  # type: ignore
            else:
                # Cosine and Step just step
                scheduler.step()  # type: ignore
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # Print epoch results
        if val_loader:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.2e}')
        else:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'LR: {current_lr:.2e}')
        
        # Save best model
        current_loss = val_loss if val_loader else train_loss
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f"💾 Saved best model: {best_loss:.4f}")
        
        # Early stopping check
        if epoch > 20 and len(history['train_loss']) > 10:
            recent_losses = history['val_loss'][-10:] if val_loader else history['train_loss'][-10:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                print("Early stopping triggered!")
                break
    
    return history

class MultipletDuoDataset(Dataset):
    """Dataset that creates multiple resampled versions (multiplets) of each sequence for data amplification.
    
    preserve_endpoints behavior:
      - If preserve_endpoints is False: resampling behaves as before (no special endpoint handling).
      - If preserve_endpoints is True: the preserved start/end points for each resample are chosen
        randomly from the first 5 and last 5 points of the original sequence respectively. The
        resampler is run without forcing endpoints, and then the first/last sample of the resampled
        vector are replaced by the chosen original endpoint values to guarantee endpoint preservation
        among the first/last 5 samples.
    """

    def __init__(
        self, 
        X_list, 
        y_name_list, 
        mech_list, 
        label_mode: LabelMode = LabelMode.CE,
        min_length: int = 8,
        max_length: int = 40,
        multiplets_per_sequence: int = 5,
        resampling_strategy: ResamplingStrategy = ResamplingStrategy.UNIFORM,
        preserve_endpoints: bool = True,
        length_distribution: str = "uniform",
        include_original: bool = True,
        use_suffix: bool = True  # NEW: Use suffix-based labels
    ):
        self.mech_list = mech_list
        self.label_mode = LabelMode(label_mode)
        self.use_suffix = use_suffix
        
        # Build class mapping with suffix support
        self.class_to_idx, self.idx_to_class, self.full_to_suffix = build_class_mapping(
            mech_list, use_suffix=use_suffix
        )
        
        self.min_length = min_length
        self.max_length = max_length
        self.multiplets_per_sequence = multiplets_per_sequence
        self.resampling_strategy = ResamplingStrategy(resampling_strategy)
        self.preserve_endpoints = preserve_endpoints
        self.length_distribution = length_distribution
        self.include_original = include_original
        
        # Generate all multiplets at initialization for consistent training
        self.multiplet_X = []
        self.multiplet_Y = []
        self.multiplet_lengths = []
        self.sequence_sources = []
        
        print(f"Generating multiplets: {multiplets_per_sequence} versions per sequence...")
        
        multiplet_count = 0
        for seq_idx, x in enumerate(X_list):
            if len(x) >= min_length:
                x_array = np.array(x, dtype=np.float32)
                original_length = len(x)
                
                # Include original sequence if requested
                if include_original:
                    if original_length <= max_length:
                        self.multiplet_X.append(torch.as_tensor(x_array, dtype=torch.float32))
                        self.multiplet_lengths.append(original_length)
                    else:
                        # Intelligently truncate original using flexible endpoints if requested
                        truncated = self._resample_with_flexible_endpoints(
                            x_array, max_length, strategy=resampling_strategy, preserve_endpoints=True
                        )
                        self.multiplet_X.append(torch.as_tensor(truncated, dtype=torch.float32))
                        self.multiplet_lengths.append(len(truncated))
                    
                    self.multiplet_Y.append(y_name_list[seq_idx])
                    self.sequence_sources.append(seq_idx)
                    multiplet_count += 1
                
                # Generate multiplets
                for multiplet_idx in range(multiplets_per_sequence):
                    target_length = self._sample_target_length()
                    
                    if target_length >= original_length:
                        resampled = x_array.copy()
                    else:
                        resampled = self._resample_with_flexible_endpoints(
                            x_array,
                            target_length,
                            strategy=resampling_strategy,
                            preserve_endpoints=preserve_endpoints
                        )
                    
                    self.multiplet_X.append(torch.as_tensor(resampled, dtype=torch.float32))
                    self.multiplet_lengths.append(len(resampled))
                    self.multiplet_Y.append(y_name_list[seq_idx])
                    self.sequence_sources.append(seq_idx)
                    multiplet_count += 1
        
        valid_originals = len([x for x in X_list if len(x) >= min_length])
        print(f"Generated {multiplet_count} total sequences from {valid_originals} originals")
        if valid_originals > 0:
            print(f"Amplification factor: {multiplet_count / valid_originals:.1f}x")
        else:
            print("Amplification factor: N/A (no valid originals)")
        
        # Convert labels - NOW USING SUFFIX MAPPING
        if self.label_mode == LabelMode.CE:
            if use_suffix and self.full_to_suffix is not None:
                # Convert full mechanism names to suffix-based labels
                self.Y_tensor = torch.tensor([
                    self.class_to_idx[self.full_to_suffix[name]] 
                    for name in self.multiplet_Y
                ], dtype=torch.long)
            else:
                # Use full mechanism names
                self.Y_tensor = torch.tensor([
                    self.class_to_idx[name] 
                    for name in self.multiplet_Y
                ], dtype=torch.long)
        else:
            K = len(self.idx_to_class)  # Use actual number of classes (suffixes)
            Y = torch.zeros((len(self.multiplet_Y), K), dtype=torch.float32)
            for i, name in enumerate(self.multiplet_Y):
                if use_suffix and self.full_to_suffix is not None:
                    suffix = self.full_to_suffix[name]
                    Y[i, self.class_to_idx[suffix]] = 1.0
                else:
                    Y[i, self.class_to_idx[name]] = 1.0
            self.Y_tensor = Y
        
        # Convert lengths to tensor
        self.lengths_tensor = torch.tensor(self.multiplet_lengths, dtype=torch.long)
        
        # Print statistics
        self._print_statistics()

    def _sample_target_length(self):
        """Sample target length according to specified distribution."""
        if self.length_distribution == "uniform":
            return np.random.randint(self.min_length, self.max_length + 1)
        elif self.length_distribution == "normal":
            mean = (self.min_length + self.max_length) / 2
            std = (self.max_length - self.min_length) / 6
            length = int(np.random.normal(mean, std))
            return int(np.clip(length, self.min_length, self.max_length))
        elif self.length_distribution == "exponential":
            scale = (self.max_length - self.min_length) / 3
            length = int(np.random.exponential(scale) + self.min_length)
            return int(np.clip(length, self.min_length, self.max_length))
        else:
            return np.random.randint(self.min_length, self.max_length + 1)

    def _choose_flexible_endpoint_indices(self, orig_len):
        """Choose start and end indices from the first 5 and last 5 samples, respectively.
        
        Guarantees start_idx < end_idx and both within [0, orig_len-1].
        """
        if orig_len <= 2:
            return 0, orig_len - 1
        
        # allowable start indices: 0 .. min(4, orig_len-2)  (ensure room for end)
        max_start = min(4, orig_len - 2)
        # allowable end indices: max(1, orig_len-5) .. orig_len-1
        min_end = max(1, orig_len - 5)
        max_end = orig_len - 1
        
        # If ranges overlap or are inverted (small sequences), fall back to safe values
        if max_start < 0:
            start_idx = 0
        else:
            start_idx = int(np.random.randint(0, max_start + 1))
        
        if min_end > max_end:
            end_idx = max_end
        else:
            end_idx = int(np.random.randint(min_end, max_end + 1))
        
        # Ensure start < end; if not, force end = start + 1 (capped by last index)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, orig_len - 1)
        
        return start_idx, end_idx

    def _resample_with_flexible_endpoints(self, x_array, target_length, strategy, preserve_endpoints):
        """Resample x_array to target_length while allowing preservation of any of the first 5
        and last 5 points as endpoints when preserve_endpoints=True.
        
        Approach:
          - If preserve_endpoints is False: call resample_sequence normally (preserve_endpoints=False).
          - If preserve_endpoints is True: choose start/end indices from the first/last 5 points,
            call resample_sequence with preserve_endpoints=False (so resampler can freely sample),
            then overwrite the first and last sample of the resampled vector with the chosen
            original endpoint values. This guarantees that the endpoints come from the first/last
            five original points while avoiding changes to the external resampling utility.
        """
        orig_len = len(x_array)
        if target_length >= orig_len:
            return x_array.copy()

        # If preserve_endpoints not requested, just call the resampler normally.
        if not preserve_endpoints:
            resampled, _ = resample_sequence(
                x_array, target_length, strategy=strategy, preserve_endpoints=False
            )
            return resampled

        # preserve_endpoints == True: choose flexible endpoints
        start_idx, end_idx = self._choose_flexible_endpoint_indices(orig_len)
        start_val = x_array[start_idx]
        end_val = x_array[end_idx]

        # Resample without forcing endpoints (so the resampler can do its job),
        # then enforce the selected endpoints by assignment.
        resampled, _ = resample_sequence(
            x_array, target_length, strategy=strategy, preserve_endpoints=False
        )

        # Ensure resampled length is at least 1 (it should be >= min_length in our usage)
        if len(resampled) == 0:
            # Fallback: create tiny array with start and/or end values
            if target_length == 1:
                resampled = np.array([start_val], dtype=x_array.dtype)
            else:
                # create array filled with start_val and set last to end_val
                resampled = np.full((target_length,), start_val, dtype=x_array.dtype)
                resampled[-1] = end_val
            return resampled

        # Replace first and last elements with the chosen endpoints
        resampled[0] = start_val
        resampled[-1] = end_val

        # If target_length == 1, we already set resampled[0] above.
        return resampled

    def _print_statistics(self):
        """Print dataset statistics."""
        lengths = self.lengths_tensor.numpy()
        print(f"\nDataset Statistics:")
        print(f"  Total sequences: {len(self)}")
        print(f"  Length range: {lengths.min()} to {lengths.max()}")
        print(f"  Mean length: {lengths.mean():.1f}")
        print(f"  Length distribution:")
        
        # Count by length
        unique_lengths, counts = np.unique(lengths, return_counts=True)
        for length, count in zip(unique_lengths, counts):
            print(f"    {length:2d} points: {count:4d} sequences ({count/len(self)*100:.1f}%)")
        
        # Count by class (show both full names and suffixes if using suffix mode)
        print(f"  Class distribution:")
        if self.use_suffix and self.full_to_suffix is not None:
            # Group by suffix
            suffix_counts = {}
            for label_name in self.multiplet_Y:
                suffix = self.full_to_suffix[label_name]
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            
            for suffix, count in sorted(suffix_counts.items()):
                print(f"    {suffix}: {count:4d} sequences ({count/len(self)*100:.1f}%)")
        else:
            # Show full names
            class_counts = {}
            for label_name in self.multiplet_Y:
                class_counts[label_name] = class_counts.get(label_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                print(f"    {class_name}: {count:4d} sequences ({count/len(self)*100:.1f}%)")

    def __len__(self):
        return len(self.multiplet_X)

    def __getitem__(self, idx):
        return self.multiplet_X[idx], self.Y_tensor[idx], self.lengths_tensor[idx]

    def get_source_info(self, idx):
        """Get information about which original sequence this multiplet came from."""
        return self.sequence_sources[idx]
    
    def get_label_name(self, idx):
        """Get the label name (suffix or full name) for a given index."""
        full_name = self.multiplet_Y[idx]
        if self.use_suffix and self.full_to_suffix is not None:
            return self.full_to_suffix[full_name]
        else:
            return full_name
 
def multiplet_collate_fn(batch):
    """Optimized collate function for multiplet sequences (similar to dynamic but for pre-generated data)."""
    sequences, labels, lengths = zip(*batch)
    
    # Find maximum length in this batch
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences to max length in batch
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.zeros((max_len - len(seq), seq.shape[1]), dtype=torch.float32)
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack everything
    x_batch = torch.stack(padded_sequences)
    length_batch = torch.stack(lengths)
    y_batch = torch.stack(labels) if labels[0].ndim > 0 else torch.tensor(labels, dtype=torch.long)
    
    return x_batch, y_batch, length_batch

def Train_Model_Data_Amplification():
    """Example using multiplet resampling for data amplification."""
    
    # Load data
    print("Loading data for multiplet amplification...")
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=0,
        end_iteration=25000,  
        min_time_points=100
    )
    
    print(f"Original dataset: {len(X_list)} sequences")
    
    # Create multiplet dataset with 8x amplification
    dataset = MultipletDuoDataset(
        X_list, 
        y_name_list, 
        mech_list,
        min_length=10,
        max_length=90,
        multiplets_per_sequence=10,  
        resampling_strategy=ResamplingStrategy.UNIFORM,
        preserve_endpoints=True,
        length_distribution="uniform",
        use_suffix= False,
        include_original=False
    )
    
    # Split data
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Training set: {len(train_dataset)} sequences")
    print(f"Validation set: {len(val_dataset)} sequences")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1536,  
        shuffle=True, 
        collate_fn=multiplet_collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=512, 
        shuffle=False, 
        collate_fn=multiplet_collate_fn
    )
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    """Pooling_method: attention and multi and last 
        Attention is overall pooling 
        multi is a combo of attention, mean and last
        last is based on final input"""

    # Create enhanced model
    model = EnhancedLSTM(
        input_size=4,
        hidden_size=256,  
        num_layers=3,
        num_classes=8,
        dropout=0.1,
        use_residual=True,
        pooling_method="multi" 
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model with cosine annealing scheduler
    history = train_simple_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1500,
        learning_rate=0.25e-3,
        device=device,
        model_name="Deact_AI_v1.0_3x256_LSTM_M1",
        scheduler_type='plateau'  
    )
    
    return model, history, dataset

def analyze_multiplet_diversity():
    """Analyze the diversity of multiplets generated from the same sequence."""
    
    # Load small dataset
    X_list, y_name_list, P_list = get_data_filtered_duos(
        mech_list=mech_list,
        start_iteration=0,
        end_iteration=100,
        min_time_points=15
    )
    
    # Take one sequence and generate multiplets
    original_seq = X_list[0]
    print(f"Original sequence length: {len(original_seq)}")
    print(f"Original sequence label: {y_name_list[0]}")
    
    # Generate multiplets
    dataset = MultipletDuoDataset(
        [original_seq], [y_name_list[0]], mech_list,
        min_length=15, max_length=75,
        multiplets_per_sequence=10,
        resampling_strategy=ResamplingStrategy.UNIFORM,
        preserve_endpoints=True,
        include_original=True
    )
    
    print(f"\nGenerated {len(dataset)} multiplets from 1 original sequence")
    
    # Analyze the multiplets
    print("\nMultiplet analysis:")
    for i in range(len(dataset)):
        seq, label, length = dataset[i]
        source = dataset.get_source_info(i)
        seq_type = "Original" if i == 0 and dataset.include_original else "Multiplet"
        print(f"  {seq_type} {i}: {length.item()} points, source sequence {source}")
    
    # Show first few points of each to demonstrate endpoint preservation
    print(f"\nEndpoint preservation check (first 3 and last 3 points):")
    original_array = np.array(original_seq)
    print(f"Original first 3: {original_array[:3, 0]}")  # First feature only
    print(f"Original last 3:  {original_array[-3:, 0]}")
    
    for i in range(min(5, len(dataset))):
        seq, _, length = dataset[i]
        seq_array = seq.numpy()
        print(f"Seq {i} first 3: {seq_array[:3, 0]}")
        print(f"Seq {i} last 3:  {seq_array[-3:, 0]}")
    
    return dataset

if __name__ == "__main__":
    Train_Model_Data_Amplification()

