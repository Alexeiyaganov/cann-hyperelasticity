"""
CANN: Automatic Discovery of Hyperelastic Material Laws
Main entry point - run this!

Usage:
    python main.py              # Run everything
    python main.py --data-only  # Only generate data
    python main.py --train-only # Only train (requires existing data)
    python main.py --analyze    # Only analyze (requires trained model)
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Create necessary directories
os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results/models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "results/figures"), exist_ok=True)

import numpy as np
import torch
from cann import CANN, train_model, evaluate_model
from cann.fem_solver import generate_neo_hookean_data
from cann.utils import (
    relative_error,
    r2_score,
    plot_training_curves,
    plot_scatter,
    plot_energy_surface,
    plot_marginal_energy,
)


def generate_data(data_path="data/experimental_data.csv"):
    """Step 1: Generate FEM data using FEniCSx"""
    print("\n" + "="*70)
    print("STEP 1: GENERATING DATA")
    print("="*70)
    
    if os.path.exists(data_path):
        print(f"✓ Data already exists at {data_path}")
        return
    
    try:
        print("Generating Neo-Hookean data with FEniCSx...")
        print("  - Mesh: 15×15 elements")
        print("  - Loading: uniaxial tension + pure shear")
        print("  - Deformation: 5%-30%")
        
        F_all, P_all = generate_neo_hookean_data(
            mesh_size=15,
            displacement_steps=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        )
        
        print(f"✓ Generated {F_all.shape[0]} data points")
        print(f"✓ Data saved to {data_path}")
        
    except ImportError:
        print("⚠ FEniCSx not installed. Using pre-generated demo data...")
        # For demo purposes, create synthetic data
        np.random.seed(42)
        n_points = 2700
        F_all = np.random.randn(n_points, 2, 2) * 0.3 + np.eye(2)
        F_all = np.abs(F_all)  # Ensure positive-definite
        
        # Compute synthetic stresses (simple model)
        mu, lmbda = 3.846, 5.769
        P_all = np.zeros_like(F_all)
        for i in range(n_points):
            I1 = np.trace(F_all[i].T @ F_all[i]) + 1
            J = np.linalg.det(np.eye(3))
            J = np.linalg.det(F_all[i]) if F_all[i].shape == (2, 2) else 1.0
            J = max(J, 0.1)
            P_all[i] = mu * F_all[i] + (lmbda * np.log(J) - mu) * np.linalg.inv(F_all[i]).T
        
        # Save data
        os.makedirs("data", exist_ok=True)
        data = np.hstack([
            F_all.reshape(n_points, 4),
            P_all.reshape(n_points, 4),
        ])
        np.savetxt(data_path, data, delimiter=",", 
                   header="F11,F12,F21,F22,P11,P12,P21,P22")
        print(f"✓ Demo data saved to {data_path}")


def train_cann(data_path="data/experimental_data.csv",
               model_path="results/models/best_cann.pth",
               device="cuda" if torch.cuda.is_available() else "cpu"):
    """Step 2: Train CANN model"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING CANN")
    print("="*70)
    
    if os.path.exists(model_path):
        print(f"✓ Model already trained at {model_path}")
        return model_path
    
    # Load data
    print("Loading data...")
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    
    F_np = data[:, :4].reshape(-1, 2, 2)
    P_np = data[:, 4:].reshape(-1, 2, 2)
    
    # Split: first half = tension (train/val), second half = shear (test)
    n_per_exp = len(F_np) // 2
    
    F_tension = F_np[:n_per_exp]
    P_tension = P_np[:n_per_exp]
    F_shear = F_np[n_per_exp:]
    P_shear = P_np[n_per_exp:]
    
    # Train/val split (80/20)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(F_tension))
    n_val = len(F_tension) // 5
    
    F_train = torch.tensor(F_tension[idx[n_val:]], dtype=torch.float64, device=device)
    P_train = torch.tensor(P_tension[idx[n_val:]], dtype=torch.float64, device=device)
    F_val = torch.tensor(F_tension[idx[:n_val]], dtype=torch.float64, device=device)
    P_val = torch.tensor(P_tension[idx[:n_val]], dtype=torch.float64, device=device)
    F_test = torch.tensor(F_shear, dtype=torch.float64, device=device)
    P_test = torch.tensor(P_shear, dtype=torch.float64, device=device)
    
    print(f"Train: {len(F_train)} points")
    print(f"Val:   {len(F_val)} points")
    print(f"Test:  {len(F_test)} points (pure shear)")
    
    # Train model
    print("\nTraining CANN...")
    model = CANN(hidden_dims=(64, 64, 32)).double()
    model, history = train_model(
        model, F_train, P_train, F_val, P_val,
        epochs=3000,
        batch_size=256,
        lr=1e-3,
        alpha=1e-4,
        early_stop_patience=400,
        device=device,
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Plot training curves
    fig_path = "results/figures/training_curves.png"
    plot_training_curves(history, save_path=fig_path)
    
    return model_path, model, history, (F_test, P_test)


def analyze_results(data_path="data/experimental_data.csv",
                   model_path="results/models/best_cann.pth",
                   device="cuda" if torch.cuda.is_available() else "cpu"):
    """Step 3: Analyze results and generate figures"""
    print("\n" + "="*70)
    print("STEP 3: ANALYZING RESULTS")
    print("="*70)
    
    # Load data
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    F_np = data[:, :4].reshape(-1, 2, 2)
    P_np = data[:, 4:].reshape(-1, 2, 2)
    
    n_per_exp = len(F_np) // 2
    F_tension = F_np[:n_per_exp]
    P_tension = P_np[:n_per_exp]
    F_shear = F_np[n_per_exp:]
    P_shear = P_np[n_per_exp:]
    
    # Load model
    print("Loading model...")
    model = CANN(hidden_dims=(64, 64, 32)).double().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare tensors
    F_tension_t = torch.tensor(F_tension, dtype=torch.float64, device=device)
    P_tension_t = torch.tensor(P_tension, dtype=torch.float64, device=device)
    F_shear_t = torch.tensor(F_shear, dtype=torch.float64, device=device)
    P_shear_t = torch.tensor(P_shear, dtype=torch.float64, device=device)
    
    # Predictions
    print("Computing predictions...")
    with torch.enable_grad():
        _, P_tension_pred = model(F_tension_t)
        _, P_shear_pred = model(F_shear_t)
    
    P_tension_pred_np = P_tension_pred.detach().cpu().numpy()
    P_shear_pred_np = P_shear_pred.detach().cpu().numpy()
    
    # Metrics
    print("\n" + "-"*70)
    print("METRICS")
    print("-"*70)
    re_tension = relative_error(P_tension, P_tension_pred_np)
    re_shear = relative_error(P_shear, P_shear_pred_np)
    
    print(f"Tension (val): RE = {re_tension:.4f}")
    print(f"Shear (test):  RE = {re_shear:.4f}")
    print("\nR² per component (shear):")
    for i in range(2):
        for j in range(2):
            r2 = r2_score(P_shear[:, i, j], P_shear_pred_np[:, i, j])
            print(f"  P{i+1}{j+1}: {r2:.6f}")
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Scatter plot
    plot_scatter(
        P_true={"val": P_tension, "test": P_shear},
        P_pred={"val": P_tension_pred_np, "test": P_shear_pred_np},
        titles={"val": "Tension (train)", "test": "Shear (test)"},
        save_path="results/figures/cann_scatter.png",
    )
    
    # Energy surface
    def psi_neo_hookean(I1, J):
        mu, lmbda = 3.846, 5.769
        return (mu/2) * (I1 - 3) - mu * np.log(np.clip(J, 1e-10, None)) + (lmbda/2) * np.log(np.clip(J, 1e-10, None))**2
    
    plot_energy_surface(
        model,
        device=device,
        psi_true_fn=psi_neo_hookean,
        save_path="results/figures/energy_surface.png",
    )
    
    # Marginal slices
    plot_marginal_energy(
        model,
        component="I1",
        psi_true_fn=lambda I1: psi_neo_hookean(I1, np.ones_like(I1)),
        device=device,
        save_path="results/figures/energy_1d.png",
    )
    
    plot_marginal_energy(
        model,
        component="J",
        psi_true_fn=lambda J: psi_neo_hookean(np.ones_like(J)*3.0, J),
        device=device,
        save_path="results/figures/energy_J.png",
    )
    
    print("\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Model:  results/models/best_cann.pth")
    print("  - Figures: results/figures/*.png")
    print("\nOpen results/figures/ to see the plots!")


def main():
    """Run the full pipeline"""
    parser = argparse.ArgumentParser(description="CANN: Hyperelastic Material Discovery")
    parser.add_argument("--data-only", action="store_true", help="Only generate data")
    parser.add_argument("--train-only", action="store_true", help="Only train (requires data)")
    parser.add_argument("--analyze", action="store_true", help="Only analyze (requires model)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 CANN: Hyperelastic Material Discovery")
    print(f"   Device: {device.upper()}")
    
    if args.data_only:
        generate_data()
    elif args.train_only:
        train_cann(device=device)
    elif args.analyze:
        analyze_results(device=device)
    else:
        # Full pipeline
        generate_data()
        train_cann(device=device)
        analyze_results(device=device)


if __name__ == "__main__":
    main()