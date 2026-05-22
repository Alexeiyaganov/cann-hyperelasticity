"""
Utility functions for metrics, visualization, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple


# ============================================================================
# METRICS
# ============================================================================

def relative_error(P_true: np.ndarray, P_pred: np.ndarray) -> float:
    """
    Compute relative error between true and predicted stresses.
    
    RE = ||P_true - P_pred||_F / ||P_true||_F
    
    Args:
        P_true: True stresses (N, 2, 2)
        P_pred: Predicted stresses (N, 2, 2)
    
    Returns:
        Relative error (scalar)
    """
    return np.linalg.norm(P_true.flatten() - P_pred.flatten()) / np.linalg.norm(
        P_true.flatten()
    )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² coefficient of determination.
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(
    history: Dict,
    save_path: str = "training_curves.png",
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history from train_model()
        save_path: Where to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(len(history["train"]))
    
    # Loss curves
    ax1.semilogy(epochs_range, history["train"], label="Train")
    if history["val"]:
        ax1.semilogy(epochs_range[: len(history["val"])], history["val"], 
                     label="Val", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training Curves")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    # Learning rate schedule
    ax2.semilogy(epochs_range[: len(history["lr"])], history["lr"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_scatter(
    P_true: Dict,
    P_pred: Dict,
    titles: Dict,
    save_path: str = "scatter.png",
) -> None:
    """
    Plot P_true vs P_pred scatter plots for all components.
    
    Args:
        P_true: Dict with 'val' and 'test' keys containing true stresses
        P_pred: Dict with 'val' and 'test' keys containing predicted stresses
        titles: Dict with descriptions for plots
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    components = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = ["P11", "P12", "P21", "P22"]
    
    for row, (key, title) in enumerate(titles.items()):
        P_t = P_true[key]
        P_p = P_pred[key]
        
        for col, ((i, j), label) in enumerate(zip(components, labels)):
            ax = axes[row, col]
            tv = P_t[:, i, j]
            pv = P_p[:, i, j]
            
            ax.scatter(tv, pv, s=4, alpha=0.35, rasterized=True)
            lims = [min(tv.min(), pv.min()), max(tv.max(), pv.max())]
            ax.plot(lims, lims, "r--", lw=1.5)
            
            r2 = r2_score(tv, pv)
            ax.set_title(f"{title}\n{label}, $R^2={r2:.4f}$", fontsize=8)
            ax.set_xlabel("True P")
            ax.set_ylabel("Predicted P")
    
    plt.suptitle("True vs Predicted Stresses", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_energy_surface(
    model: torch.nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    psi_true_fn=None,  # Optional: function to compute true energy
    save_path: str = "energy_surface.png",
) -> None:
    """
    Plot energy surface ψ(I1, J) predicted by CANN.
    
    Args:
        model: Trained CANN model
        device: torch device
        psi_true_fn: Optional function(I1, J) -> psi_true for comparison
        save_path: Where to save the figure
    """
    model.eval()
    model.to(device)
    
    # Create grid
    I1_lin = np.linspace(2.6, 5.5, 80)
    J_lin = np.linspace(0.75, 1.30, 80)
    I1_grid, J_grid = np.meshgrid(I1_lin, J_lin)
    
    # Compute CANN energy
    I1_t = torch.tensor(I1_grid.ravel(), dtype=torch.float64, device=device)
    lnJ_t = torch.tensor(np.log(J_grid.ravel()), dtype=torch.float64, device=device)
    
    with torch.no_grad():
        psi_raw = model._psi_raw(I1_t, lnJ_t)
        psi_ref = model._psi_raw(torch.full_like(I1_t, 3.0), torch.zeros_like(lnJ_t))
        psi_cann = (psi_raw - psi_ref).cpu().numpy().reshape(I1_grid.shape)
    
    # Setup figure
    if psi_true_fn is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        psi_true = psi_true_fn(I1_grid, J_grid)
        levels = np.linspace(
            min(psi_true.min(), psi_cann.min()),
            max(psi_true.max(), psi_cann.max()),
            25,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        levels = 20
    
    # Plot CANN
    im = axes[0].contourf(I1_grid, J_grid, psi_cann, levels=levels, cmap="viridis")
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel("$I_1$")
    axes[0].set_ylabel("$J$")
    axes[0].set_title("CANN: $\\hat{\\psi}(I_1, J)$")
    
    if psi_true_fn is not None:
        # Plot true
        im = axes[1].contourf(I1_grid, J_grid, psi_true, levels=levels, cmap="viridis")
        plt.colorbar(im, ax=axes[1])
        axes[1].set_xlabel("$I_1$")
        axes[1].set_ylabel("$J$")
        axes[1].set_title("True: $\\psi(I_1, J)$")
        
        # Plot error
        im = axes[2].contourf(
            I1_grid, J_grid, np.abs(psi_cann - psi_true),
            levels=20, cmap="Reds"
        )
        plt.colorbar(im, ax=axes[2])
        axes[2].set_xlabel("$I_1$")
        axes[2].set_ylabel("$J$")
        axes[2].set_title("$|\\hat{\\psi} - \\psi|$")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def plot_marginal_energy(
    model: torch.nn.Module,
    component: str = "I1",  # "I1" or "J"
    psi_true_fn=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "marginal.png",
) -> None:
    """
    Plot marginal slice of energy surface.
    
    Args:
        model: Trained CANN model
        component: "I1" for ψ(I1)|J=1, "J" for ψ(J)|I1=3
        psi_true_fn: Optional function for true energy
        device: torch device
        save_path: Where to save the figure
    """
    model.eval()
    model.to(device)
    
    if component == "I1":
        # ψ(I1) at J=1
        I1_vals = np.linspace(2.5, 6.5, 200)
        lnJ_vals = np.zeros(200)
        xlabel, title = "$I_1$", "Energy at $J=1$ (deviatoric deformation)"
    else:  # J
        # ψ(J) at I1=3
        J_vals = np.linspace(0.6, 1.5, 200)
        I1_vals = np.ones(200) * 3.0
        lnJ_vals = np.log(J_vals)
        xlabel, title = "$J$", "Energy at $I_1=3$ (volumetric deformation)"
    
    # Compute energy
    I1_t = torch.tensor(I1_vals, dtype=torch.float64, device=device)
    lnJ_t = torch.tensor(lnJ_vals, dtype=torch.float64, device=device)
    
    with torch.no_grad():
        psi_raw = model._psi_raw(I1_t, lnJ_t)
        psi_ref = model._psi_raw(torch.full_like(I1_t, 3.0), torch.zeros_like(lnJ_t))
        psi_cann = (psi_raw - psi_ref).cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(I1_vals if component == "I1" else J_vals, psi_cann, 
            "b-", lw=2.5, label="CANN")
    
    if psi_true_fn is not None:
        psi_true = psi_true_fn(I1_vals if component == "I1" else J_vals)
        ax.plot(I1_vals if component == "I1" else J_vals, psi_true,
                "r--", lw=2.0, label="True")
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("$\\psi$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()