"""
Training loop and evaluation utilities for CANN.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict


def train_model(
    model: nn.Module,
    F_train: torch.Tensor,
    P_train: torch.Tensor,
    F_val: Optional[torch.Tensor] = None,
    P_val: Optional[torch.Tensor] = None,
    epochs: int = 3000,
    batch_size: int = 256,
    lr: float = 1e-3,
    alpha: float = 1e-4,
    early_stop_patience: int = 400,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, Dict]:
    """
    Train CANN model.
    
    Args:
        model: CANN instance
        F_train: Training deformation gradients (N, 2, 2)
        P_train: Training stresses (N, 2, 2)
        F_val: Validation deformation gradients (optional)
        P_val: Validation stresses (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        alpha: Regularization weight for ψ(I) normalization
        early_stop_patience: Patience for early stopping
        device: torch device
    
    Returns:
        model: Trained model
        history: Training history dict with keys 'train', 'val', 'lr'
    """
    
    model = model.to(device)
    model.double()  # Use float64 for numerical stability
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=120,
        factor=0.5,
        min_lr=5e-7,
    )
    
    # Create data loader
    train_loader = DataLoader(
        TensorDataset(F_train, P_train),
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Training history
    history = {"train": [], "val": [], "lr": []}
    best_val = float("inf")
    patience_cnt = 0
    
    # Training loop
    print(f"Starting training on {device}...")
    t0 = time.time()
    
    for epoch in range(epochs):
        # === Training phase ===
        model.train()
        epoch_loss = 0.0
        
        for F_batch, P_batch in train_loader:
            F_batch, P_batch = F_batch.to(device), P_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            psi, P_pred = model(F_batch)
            
            # Loss function
            stress_loss = ((P_pred - P_batch) ** 2).mean()
            reg_loss = alpha * (psi ** 2).mean()
            loss = stress_loss + reg_loss
            
            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        history["train"].append(epoch_loss)
        
        # === Validation phase ===
        model.eval()
        if F_val is not None and P_val is not None:
            F_val = F_val.to(device)
            P_val = P_val.to(device)
            
            with torch.enable_grad():
                psi_val, P_val_pred = model(F_val)
                val_stress_loss = ((P_val_pred - P_val) ** 2).mean()
            
            history["val"].append(val_stress_loss.item())
            scheduler.step(val_stress_loss.item())
            
            # Early stopping
            if val_stress_loss.item() < best_val:
                best_val = val_stress_loss.item()
                torch.save(model.state_dict(), "best_cann.pth")
                patience_cnt = 0
            else:
                patience_cnt += 1
            
            if patience_cnt >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        history["lr"].append(optimizer.param_groups[0]["lr"])
        
        # === Logging ===
        if (epoch + 1) % 300 == 0 or epoch == epochs - 1:
            val_str = (
                f" | val={history['val'][-1]:.3e}"
                if history["val"]
                else ""
            )
            print(
                f"  Epoch {epoch:4d} | train={epoch_loss:.3e}{val_str} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e}"
            )
    
    t1 = time.time()
    print(f"\nTraining completed in {t1-t0:.1f}s")
    
    # Load best model
    if F_val is not None:
        model.load_state_dict(torch.load("best_cann.pth"))
    
    return model, history


def evaluate_model(
    model: nn.Module,
    F_test: torch.Tensor,
    P_test: torch.Tensor,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained CANN model
        F_test: Test deformation gradients (N, 2, 2)
        P_test: Test stresses (N, 2, 2)
        device: torch device
    
    Returns:
        metrics: Dictionary with 'RE' and 'R2' keys
    """
    model.eval()
    model.to(device)
    
    F_test = F_test.to(device).double()
    P_test = P_test.to(device).double()
    
    with torch.enable_grad():
        _, P_pred = model(F_test)
    
    P_pred_np = P_pred.detach().cpu().numpy()
    P_test_np = P_test.detach().cpu().numpy()
    
    # Relative error
    re = np.linalg.norm(P_test_np.flatten() - P_pred_np.flatten()) / np.linalg.norm(
        P_test_np.flatten()
    )
    
    # R² scores per component
    r2_per_component = {}
    for i in range(2):
        for j in range(2):
            y_true = P_test_np[:, i, j]
            y_pred = P_pred_np[:, i, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            r2_per_component[f"P{i+1}{j+1}"] = r2
    
    r2_mean = np.mean(list(r2_per_component.values()))
    
    return {
        "RE": re,
        "R2_mean": r2_mean,
        "R2": r2_per_component,
        "P_pred": P_pred_np,
    }