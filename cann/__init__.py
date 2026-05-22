"""
CANN: Constitutive Artificial Neural Networks for Hyperelasticity

A package for automatically discovering hyperelastic material laws using 
physics-informed neural networks.

Example:
    >>> from cann import CANN, train_model
    >>> model = CANN(hidden_dims=(64, 64, 32))
    >>> F_train, P_train = load_data()
    >>> model = train_model(model, F_train, P_train)
    >>> P_pred = model(F_test)[1]
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

from .model import CANN
from .training import train_model, evaluate_model
from .utils import (
    relative_error,
    r2_score,
    plot_training_curves,
    plot_scatter,
    plot_energy_surface,
)

__all__ = [
    "CANN",
    "train_model",
    "evaluate_model",
    "relative_error",
    "r2_score",
    "plot_training_curves",
    "plot_scatter",
    "plot_energy_surface",
]