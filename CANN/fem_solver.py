"""
FEM solver for generating hyperelastic material data using FEniCSx.

This module handles:
1. Mesh creation
2. Neo-Hookean constitutive model
3. Nonlinear FEM solution via Newton's method
4. Data extraction (F, P) at element centers
"""

import numpy as np
import warnings

try:
    import ufl
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import NonlinearProblem
    from dolfinx.nls.petsc import NewtonSolver
    from mpi4py import MPI
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    warnings.warn("FEniCSx not available. Install with official installer from fem-on-colab.github.io")


def run_experiment(displacement_value, mesh_size=10, experiment_type="tension"):
    """
    Run a single FEM simulation (uniaxial tension or pure shear).
    
    Args:
        displacement_value: Magnitude of applied displacement
        mesh_size: Number of elements along each direction
        experiment_type: "tension" or "shear"
    
    Returns:
        F_values: Deformation gradients (n_elements, 2, 2)
        P_values: First Piola-Kirchhoff stresses (n_elements, 2, 2)
    """
    
    if not FENICS_AVAILABLE:
        raise ImportError("FEniCSx required for FEM data generation")
    
    # === 1. MESH ===
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_size, mesh_size)
    V = fem.functionspace(domain, ("CG", 1, (2,)))
    
    # === 2. BOUNDARY CONDITIONS ===
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.logical_and(
            np.isclose(x[0], 1.0),
            np.logical_and(x[1] >= 0.33, x[1] <= 0.66),
        )
    
    left_dofs = fem.locate_dofs_geometrical(V, left_boundary)
    right_dofs = fem.locate_dofs_geometrical(V, right_boundary)
    
    u_left = np.array([0.0, 0.0], dtype=default_scalar_type)
    
    if experiment_type == "tension":
        u_right = np.array([displacement_value, 0.0], dtype=default_scalar_type)
    elif experiment_type == "shear":
        u_right = np.array([0.0, displacement_value], dtype=default_scalar_type)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    bc_left = fem.dirichletbc(u_left, left_dofs, V)
    bc_right = fem.dirichletbc(u_right, right_dofs, V)
    bcs = [bc_left, bc_right]
    
    # === 3. VARIATIONAL FORMULATION ===
    u = fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)
    
    # Kinematics
    I = ufl.Identity(domain.geometry.dim)
    F_def = ufl.variable(I + ufl.grad(u))
    J_det = ufl.det(F_def)
    C = F_def.T * F_def
    I1 = ufl.tr(C)
    
    # Neo-Hookean energy
    mu_val = 3.846
    lmbda_val = 5.769
    psi = (mu_val / 2) * (I1 - 2) - mu_val * ufl.ln(J_det) + (lmbda_val / 2) * (ufl.ln(J_det)) ** 2
    
    # First Piola-Kirchhoff stress
    P = ufl.diff(psi, F_def)
    
    # Residual and Jacobian
    Pi = psi * ufl.dx
    F_res = ufl.derivative(Pi, u, v)
    J_form = ufl.derivative(F_res, u, du)
    
    # === 4. NONLINEAR SOLVE ===
    petsc_options = {
        "snes_type": "newtonls",
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_max_it": 50,
    }
    
    problem = NonlinearProblem(
        F_res, u, bcs=bcs, J=J_form,
        petsc_options_prefix="nls_",
        petsc_options=petsc_options,
    )
    
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    solver.solve(u)
    
    # === 5. DATA EXTRACTION ===
    V_tensor = fem.functionspace(domain, ("DG", 0, (2, 2)))
    
    F_proj = fem.Function(V_tensor)
    P_proj = fem.Function(V_tensor)
    
    F_expr = fem.Expression(F_def, V_tensor.element.interpolation_points)
    P_expr = fem.Expression(P, V_tensor.element.interpolation_points)
    
    F_proj.interpolate(F_expr)
    P_proj.interpolate(P_expr)
    
    F_values = F_proj.x.array.reshape(-1, 2, 2)
    P_values = P_proj.x.array.reshape(-1, 2, 2)
    
    return F_values, P_values


def generate_neo_hookean_data(
    mesh_size=15,
    displacement_steps=None,
    save_path="data/experimental_data.csv",
):
    """
    Generate full dataset: uniaxial tension + pure shear.
    
    Args:
        mesh_size: Elements per dimension (15×15 = 225 elements)
        displacement_steps: List of deformation magnitudes
        save_path: Where to save CSV
    
    Returns:
        F_total: All deformation gradients (N, 2, 2)
        P_total: All stresses (N, 2, 2)
    """
    
    if displacement_steps is None:
        displacement_steps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    all_F = []
    all_P = []
    
    # Uniaxial tension
    print("Generating uniaxial tension data...")
    for disp in displacement_steps:
        F, P = run_experiment(disp, mesh_size, "tension")
        all_F.append(F)
        all_P.append(P)
    
    # Pure shear
    print("Generating pure shear data...")
    for disp in displacement_steps:
        F, P = run_experiment(disp, mesh_size, "shear")
        all_F.append(F)
        all_P.append(P)
    
    F_total = np.concatenate(all_F, axis=0)
    P_total = np.concatenate(all_P, axis=0)
    
    # Save
    csv_data = []
    for i in range(F_total.shape[0]):
        row = [
            F_total[i, 0, 0], F_total[i, 0, 1],
            F_total[i, 1, 0], F_total[i, 1, 1],
            P_total[i, 0, 0], P_total[i, 0, 1],
            P_total[i, 1, 0], P_total[i, 1, 1],
        ]
        csv_data.append(row)
    
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    np.savetxt(
        save_path,
        csv_data,
        delimiter=",",
        header="F11,F12,F21,F22,P11,P12,P21,P22",
        comments="",
    )
    
    print(f"✓ Saved {F_total.shape[0]} data points to {save_path}")
    
    return F_total, P_total