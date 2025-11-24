import os
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

def load_training_history_from_checkpoint(checkpoint_path: str) -> Dict:
    """
    Load training history from a PyTorch checkpoint file.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        
    Returns:
        Dictionary containing training history if available
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            # Look for common training history keys
            history_keys = ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies', 
                          'training_history', 'history', 'metrics', 'logs']
            
            for key in history_keys:
                if key in checkpoint:
                    print(f"Found training history in key: {key}")
                    return checkpoint[key]
            
            # Check if history is nested in other keys
            for key, value in checkpoint.items():
                if isinstance(value, dict) and any(hist_key in value for hist_key in ['loss', 'accuracy', 'epoch']):
                    print(f"Found training history nested in: {key}")
                    return value
            
            # Show available keys to help debug
            print("Available keys in checkpoint:")
            for key in checkpoint.keys():
                print(f"  - {key}: {type(checkpoint[key])}")
                
            return {}
        else:
            print("Checkpoint contains only model weights, no training history")
            return {}
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}

def load_training_history_from_logs(log_path: str, log_format: str = 'json') -> Dict:
    """
    Load training history from log files.
    
    Args:
        log_path: Path to log file
        log_format: Format of log file ('json', 'csv', 'pickle')
        
    Returns:
        Dictionary containing training history
    """
    try:
        if log_format == 'json':
            with open(log_path, 'r') as f:
                return json.load(f)
        elif log_format == 'csv':
            df = pd.read_csv(log_path)
            return df.to_dict('list')
        elif log_format == 'pickle':
            with open(log_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported log format: {log_format}")
            
    except Exception as e:
        print(f"Error loading logs: {e}")
        return {}

def extract_metrics_from_history(history: Dict) -> Tuple[List, List, List, List]:
    """
    Extract train/val losses and accuracies from history dictionary.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Tuple of (train_losses, val_losses, train_accuracies, val_accuracies)
    """
    # Try different possible key names
    loss_keys = ['train_loss', 'train_losses', 'training_loss', 'loss']
    val_loss_keys = ['val_loss', 'val_losses', 'validation_loss', 'valid_loss']
    acc_keys = ['train_acc', 'train_accuracy', 'train_accuracies', 'training_accuracy']
    val_acc_keys = ['val_acc', 'val_accuracy', 'val_accuracies', 'validation_accuracy', 'valid_accuracy']
    
    def find_metric(keys_list, history_dict):
        for key in keys_list:
            if key in history_dict:
                values = history_dict[key]
                # Convert to list if needed
                if isinstance(values, (np.ndarray, torch.Tensor)):
                    values = values.tolist()
                elif not isinstance(values, list):
                    values = [values]
                return values
        return []
    
    train_losses = find_metric(loss_keys, history)
    val_losses = find_metric(val_loss_keys, history)
    train_accs = find_metric(acc_keys, history)
    val_accs = find_metric(val_acc_keys, history)
    
    return train_losses, val_losses, train_accs, val_accs

def plot_training_history(
    history: Dict,
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    manual_epoch: int = 750,
    manual_train_acc: Optional[float] = 90.99,
    manual_val_acc: Optional[float] = 88.90,
    show_manual_point: bool = False,  # default False, but ignored — manual points are NOT plotted
    annotate_final_from_history: bool = True,
) -> None:
    """
    Plot training history with loss and accuracy curves.

    NOTE: This version does NOT add the manual accuracy points to the accuracy plot.
    The vertical dashed line at `manual_epoch` is still drawn so you can see that epoch
    position on the plots. Final metrics from the history are still optionally annotated.

    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save the plot (optional)
        figsize: Figure size
        manual_epoch: Epoch number to mark manually (vertical line)
        manual_train_acc: Manual training accuracy (kept for compatibility but not plotted)
        manual_val_acc: Manual validation accuracy (kept for compatibility but not plotted)
        show_manual_point: kept for backward compatibility but ignored (manual points are not shown)
        annotate_final_from_history: Whether to annotate the final accuracy from history.
    """
    train_losses, val_losses, train_accs, val_accs = extract_metrics_from_history(history)

    # Determine how many subplots we need
    has_loss = len(train_losses) > 0 or len(val_losses) > 0
    has_acc = len(train_accs) > 0 or len(val_accs) > 0

    if not has_loss and not has_acc:
        print("No training metrics found to plot")
        print("Available keys in history:")
        for key in history.keys():
            print(f"  - {key}")
        return

    # Create subplots
    n_plots = int(has_loss) + int(has_acc)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Normalize axes to a list for easy indexing
    if n_plots == 1:
        axes = [axes]
    else:
        axes = list(axes)

    plot_idx = 0

    # Plot loss
    if has_loss:
        ax = axes[plot_idx]

        if len(train_losses) > 0:
            train_epochs = list(range(1, len(train_losses) + 1))
            ax.plot(train_epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

        if len(val_losses) > 0:
            val_epochs = list(range(1, len(val_losses) + 1))
            ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        # Mark the manual epoch with a vertical dashed line (always visible)
        max_epoch_loss = 0
        if len(train_losses) > 0:
            max_epoch_loss = max(max_epoch_loss, len(train_losses))
        if len(val_losses) > 0:
            max_epoch_loss = max(max_epoch_loss, len(val_losses))
        ax.axvline(manual_epoch, color='gray', linestyle='--', alpha=0.6)
        if manual_epoch > max_epoch_loss:
            # extend x-axis so the vertical line is visible
            ax.set_xlim(right=manual_epoch + 5)

        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_idx += 1

    # Plot accuracy
    if has_acc:
        ax = axes[plot_idx]

        if len(train_accs) > 0:
            train_epochs = list(range(1, len(train_accs) + 1))
            ax.plot(train_epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2, marker='o', markevery=max(1, len(train_accs)//20))

        if len(val_accs) > 0:
            val_epochs = list(range(1, len(val_accs) + 1))
            ax.plot(val_epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='o', markevery=max(1, len(val_accs)//20))

        # Determine epoch range to ensure manual epoch is visible
        max_epoch_acc = 0
        if len(train_accs) > 0:
            max_epoch_acc = max(max_epoch_acc, len(train_accs))
        if len(val_accs) > 0:
            max_epoch_acc = max(max_epoch_acc, len(val_accs))

        ax.axvline(manual_epoch, color='gray', linestyle='--', alpha=0.6, zorder=0)
        if manual_epoch > max_epoch_acc:
            ax.set_xlim(right=manual_epoch + 5)

        # Helper: determine whether history accuracies are on a 0-1 scale or 0-100 scale
        def _history_scale() -> float:
            all_accs = []
            if len(train_accs) > 0:
                all_accs.extend(train_accs)
            if len(val_accs) > 0:
                all_accs.extend(val_accs)
            if not all_accs:
                # No history accuracies -> assume percent (0-100)
                return 100.0
            if max(all_accs) <= 1.0:
                return 1.0
            return 100.0

        hist_scale = _history_scale()

        # Normalize manual inputs to match plotting scale (kept for compatibility but not used)
        def _normalize_manual(val: Optional[float]) -> Optional[float]:
            if val is None:
                return None
            if hist_scale == 1.0:
                return val / 100.0 if val > 1.0 else val
            else:
                return val * 100.0 if val <= 1.0 else val

        manual_train_plot = _normalize_manual(manual_train_acc)
        manual_val_plot = _normalize_manual(manual_val_acc)

        # -- MANUAL POINTS ARE NO LONGER PLOTTED OR ANNOTATED --
        # The code that previously drew `manual_train_plot` and `manual_val_plot`
        # and their annotations has been removed.

        # Annotate final accuracy values from history if requested
        if annotate_final_from_history:
            if len(train_accs) > 0:
                last_epoch = len(train_accs)
                last_val = train_accs[-1]
                ax.scatter([last_epoch], [last_val], color='navy', s=40, zorder=6)
                text = f"Train final ({last_epoch}): { (f'{last_val*100:.2f}%' if hist_scale==1.0 else f'{last_val:.2f}%') }"
                ax.annotate(text,
                            xy=(last_epoch, last_val),
                            xytext=(-60, 10),
                            textcoords='offset points', fontsize=9, color='navy',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

            if len(val_accs) > 0:
                last_epoch = len(val_accs)
                last_val = val_accs[-1]
                ax.scatter([last_epoch], [last_val], color='darkred', s=40, zorder=6)
                text = f"Val final ({last_epoch}): { (f'{last_val*100:.2f}%' if hist_scale==1.0 else f'{last_val:.2f}%') }"
                ax.annotate(text,
                            xy=(last_epoch, last_val),
                            xytext=(-60, -18),
                            textcoords='offset points', fontsize=9, color='darkred',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        else:
            # When not annotating final from history, do not annotate manual values either.
            # This block intentionally left blank to avoid adding any manual annotations.
            pass

        ax.set_title('Training and Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis to 0-1 or 0-100 range for accuracy depending on history scale
        if hist_scale == 1.0:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 100)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

def analyze_training_metrics(history: Dict) -> Dict:
    """
    Analyze training metrics and provide summary statistics.
    
    Args:
        history: Training history dictionary
        
    Returns:
        Dictionary with analysis results
    """
    train_losses, val_losses, train_accs, val_accs = extract_metrics_from_history(history)
    
    analysis = {}
    
    # Loss analysis
    if len(train_losses) > 0:
        analysis['train_loss'] = {
            'final': train_losses[-1],
            'min': min(train_losses),
            'max': max(train_losses),
            'improvement': train_losses[0] - train_losses[-1] if len(train_losses) > 1 else 0
        }
    
    if len(val_losses) > 0:
        analysis['val_loss'] = {
            'final': val_losses[-1],
            'min': min(val_losses),
            'max': max(val_losses),
            'best_epoch': val_losses.index(min(val_losses)) + 1,
            'improvement': val_losses[0] - val_losses[-1] if len(val_losses) > 1 else 0
        }
    
    # Accuracy analysis
    if len(train_accs) > 0:
        analysis['train_acc'] = {
            'final': train_accs[-1],
            'max': max(train_accs),
            'min': min(train_accs),
            'improvement': train_accs[-1] - train_accs[0] if len(train_accs) > 1 else 0
        }
    
    if len(val_accs) > 0:
        analysis['val_acc'] = {
            'final': val_accs[-1],
            'max': max(val_accs),
            'min': min(val_accs),
            'best_epoch': val_accs.index(max(val_accs)) + 1,
            'improvement': val_accs[-1] - val_accs[0] if len(val_accs) > 1 else 0
        }
    
    # Overfitting analysis
    if len(train_losses) > 0 and len(val_losses) > 0:
        # Look for divergence between train and val loss
        final_gap = abs(train_losses[-1] - val_losses[-1])
        analysis['overfitting'] = {
            'final_loss_gap': final_gap,
            'potentially_overfitting': final_gap > 0.5  # Threshold can be adjusted
        }
    
    return analysis

def print_training_summary(analysis: Dict) -> None:
    """Print a summary of training metrics."""
    print("=== TRAINING SUMMARY ===")
    
    for metric_type, metrics in analysis.items():
        if metric_type == 'overfitting':
            print(f"\nOverfitting Analysis:")
            print(f"  Final loss gap: {metrics['final_loss_gap']:.4f}")
            print(f"  Potentially overfitting: {metrics['potentially_overfitting']}")
        else:
            print(f"\n{metric_type.replace('_', ' ').title()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

def visualize_model_training(
    checkpoint_path: str,
    log_path: Optional[str] = None,
    title: Optional[str] = None,
    save_plot: Optional[str] = None
) -> Dict:
    """
    Complete function to visualize training history from checkpoint or logs.
    
    Args:
        checkpoint_path: Path to model checkpoint
        log_path: Optional path to separate log file
        title: Plot title
        save_plot: Path to save plot
        
    Returns:
        Analysis dictionary
    """
    print(f"Loading training history from: {checkpoint_path}")
    
    # Try to load from checkpoint first
    history = load_training_history_from_checkpoint(checkpoint_path)
    
    # If no history in checkpoint, try log file
    if not history and log_path and os.path.exists(log_path):
        print(f"Loading from log file: {log_path}")
        history = load_training_history_from_logs(log_path)
    
    if not history:
        print("No training history found. Make sure your training script saves metrics.")
        return {}
    
    # Generate title if not provided
    if title is None:
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        title = f"Training History - {model_name}"
    
    # Plot the history
    plot_training_history(history, title=title, save_path=save_plot)
    
    # Analyze metrics
    analysis = analyze_training_metrics(history)
    print_training_summary(analysis)
    
    return analysis

# ==================== EXAMPLE USAGE ====================

def example_usage():
    """Example of how to use the visualization functions."""
    
    # Example 1: Visualize from checkpoint
    checkpoint_path = "/Users/dylanpyle/VsCode/checkpoints/EnhancedLSTM_200x3xReLUxresidxfullxmulti_best.pth"
    
    analysis = visualize_model_training(
        checkpoint_path=checkpoint_path,
        title="Enhanced LSTM Training History",
        save_plot="training_history.png"
    )
    
    # Example 2: If you have separate log files
    # analysis = visualize_model_training(
    #     checkpoint_path=checkpoint_path,
    #     log_path="training_logs.json",
    #     title="Model Training History"
    # )

if __name__ == "__main__":
    # Quick test with your checkpoint
    checkpoint_path = "/Users/dylanpyle/VsCode/checkpoints/Deact_AI_v1.0_3x256_LSTM_Full_best.pth"
    
    if os.path.exists(checkpoint_path):
        visualize_model_training(checkpoint_path)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the path to your actual checkpoint file.")

