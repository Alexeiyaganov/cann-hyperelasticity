"""
CANN Model: Constitutive Artificial Neural Network

Architecture:
    F (2×2) → I1, ln(J) → [64,64,32] NN → ψ → autograd → P = ∂ψ/∂F
"""

import torch
import torch.nn as nn


class CANN(nn.Module):
    """
    Constitutive Artificial Neural Network for hyperelasticity.
    
    The network models the strain energy density ψ(I1, J) from which
    stresses are derived via automatic differentiation: P = ∂ψ/∂F
    
    This guarantees:
    - Thermodynamic consistency (existence of energy potential)
    - Material objectivity (invariance under rigid rotations)
    - Smooth stress-strain response
    
    Args:
        hidden_dims (tuple): Dimensions of hidden layers. Default: (64, 64, 32)
    
    Example:
        >>> model = CANN(hidden_dims=(64, 64, 32))
        >>> F = torch.randn(32, 2, 2)  # Batch of deformation gradients
        >>> psi, P = model(F)          # Energy and stresses
        >>> loss = ((P - P_true) ** 2).mean()
    """

    def __init__(self, hidden_dims: tuple = (64, 64, 32)):
        """Initialize CANN architecture."""
        super().__init__()
        
        # Build sequential network with Softplus activations
        layers = []
        in_dim = 2  # Input: (I1-2, ln J)
        
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.Softplus(beta=10),
            ]
            in_dim = hidden_dim
        
        # Output layer (single scalar: energy ψ)
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        self._xavier_init()

    def _xavier_init(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _psi_raw(
        self,
        I1: torch.Tensor,
        lnJ: torch.Tensor
    ) -> torch.Tensor:
        """
        Raw energy output before normalization.
        
        Args:
            I1: First strain invariant (batch,)
            lnJ: Logarithm of Jacobian determinant (batch,)
            
        Returns:
            ψ_raw: Raw energy prediction (batch,)
        """
        # Stack features: [(I1-2), ln(J)]
        # Centering at I1=3 (undistorted state) improves numerical stability
        x = torch.stack([I1 - 2.0, lnJ], dim=-1)
        return self.net(x).squeeze(-1)

    def forward(self, F_batch: torch.Tensor) -> tuple:
        """
        Forward pass: compute energy and stresses from deformation gradient.
        
        Args:
            F_batch: Deformation gradient tensors (batch_size, 2, 2)
                     For plane strain: F = [[F11, F12],
                                           [F21, F22]]
                     with implicit F33 = 1
        
        Returns:
            psi: Strain energy density (batch_size,)
                 Normalized such that ψ(I) = 0
            P: First Piola-Kirchhoff stress (batch_size, 2, 2)
               Computed via autograd: P = ∂ψ/∂F
        """
        # Clone and mark for gradient computation
        F = F_batch.clone().requires_grad_(True)

        # === Compute invariants for plane strain (F33 = 1) ===
        # I1 = tr(F^T F) in 3D = sum(F_2x2^2) + 1
        I1 = (F * F).sum(dim=(-1, -2)) + 1.0
        
        # J = det(F) in 2D plane strain
        J = torch.linalg.det(F)
        
        # Logarithm with numerical stability
        lnJ = torch.log(J.clamp(min=1e-10))

        # === Compute normalized energy ===
        # Raw energy
        psi_raw = self._psi_raw(I1, lnJ)
        
        # Reference energy (at undistorted state: I1=3, J=1)
        psi_ref = self._psi_raw(
            torch.full_like(I1, 3.0),
            torch.zeros_like(lnJ)
        )
        
        # Normalize so ψ(I) = 0
        psi = psi_raw - psi_ref

        # === Compute stresses via automatic differentiation ===
        # P = ∂ψ/∂F (First Piola-Kirchhoff stress)
        P = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=F,
            create_graph=True,  # Needed for second derivatives (tangent stiffness)
            retain_graph=True
        )[0]

        return psi, P

    def get_tangent_stiffness(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute tangent stiffness tensor C = ∂²ψ/∂F².
        
        This is computationally expensive but useful for checking
        physical validity (must be positive definite).
        
        Args:
            F: Deformation gradient (batch_size, 2, 2)
        
        Returns:
            C: Tangent stiffness (batch_size, 2, 2, 2, 2)
        """
        psi, P = self.forward(F)
        
        # This requires second-order autograd - proceed carefully
        C = torch.zeros(
            F.shape[0], 2, 2, 2, 2,
            dtype=F.dtype,
            device=F.device
        )
        
        for i in range(2):
            for j in range(2):
                if P[0, i, j].requires_grad:
                    grad_P = torch.autograd.grad(
                        P[:, i, j].sum(),
                        F,
                        create_graph=False,
                        retain_graph=True
                    )[0]
                    C[:, i, j, :, :] = grad_P
        
        return C

    def __repr__(self) -> str:
        """String representation of the model."""
        layers_str = ", ".join([
            str(m) for m in self.net 
            if isinstance(m, nn.Linear)
        ])
        return f"CANN(layers=[2→{layers_str}→1])"