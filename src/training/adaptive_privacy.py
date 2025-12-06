"""
Adaptive Differential Privacy Mechanisms for Certified Unlearning
Implements adaptive privacy budget allocation and noise scheduling
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import optax


@dataclass
class PrivacyState:
    """Tracks privacy expenditure over time"""
    epsilon_spent: float
    delta_spent: float
    step: int
    epsilon_budget: float
    delta_budget: float
    
    def update(self, epsilon_delta: Tuple[float, float], step: int):
        """Update privacy state with new expenditure"""
        new_eps, new_delta = epsilon_delta
        return PrivacyState(
            epsilon_spent=self.epsilon_spent + new_eps,
            delta_spent=self.delta_spent + new_delta,
            step=step,
            epsilon_budget=self.epsilon_budget,
            delta_budget=self.delta_budget,
        )
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (
            self.epsilon_budget - self.epsilon_spent,
            self.delta_budget - self.delta_spent,
        )
    
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted"""
        return (
            self.epsilon_spent >= self.epsilon_budget or
            self.delta_spent >= self.delta_budget
        )


class AdaptivePrivacyMechanism:
    """Base class for adaptive privacy mechanisms"""
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
    ):
        self.epsilon_budget = epsilon_budget
        self.delta_budget = delta_budget
        self.total_steps = total_steps
        
        self.privacy_state = PrivacyState(
            epsilon_spent=0.0,
            delta_spent=0.0,
            step=0,
            epsilon_budget=epsilon_budget,
            delta_budget=delta_budget,
        )
    
    def get_noise_multiplier(self, step: int) -> float:
        """Get noise multiplier for current step"""
        raise NotImplementedError
    
    def update_privacy(self, step: int, sensitivity: float):
        """Update privacy accounting after a step"""
        raise NotImplementedError


class UniformPrivacyAllocation(AdaptivePrivacyMechanism):
    """
    Uniform privacy budget allocation across all steps
    Divides total budget equally across training steps
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        sensitivity: float = 1.0,
    ):
        super().__init__(epsilon_budget, delta_budget, total_steps)
        self.sensitivity = sensitivity
        
        # Compute uniform noise multiplier
        epsilon_per_step = epsilon_budget / total_steps
        delta_per_step = delta_budget / total_steps
        
        # From Gaussian mechanism: sigma >= sqrt(2 * ln(1.25/delta)) * sensitivity / epsilon
        self.noise_multiplier = np.sqrt(
            2 * np.log(1.25 / delta_per_step)
        ) * sensitivity / epsilon_per_step
    
    def get_noise_multiplier(self, step: int) -> float:
        return self.noise_multiplier
    
    def update_privacy(self, step: int, sensitivity: float):
        epsilon_per_step = self.epsilon_budget / self.total_steps
        delta_per_step = self.delta_budget / self.total_steps
        
        self.privacy_state = self.privacy_state.update(
            (epsilon_per_step, delta_per_step),
            step
        )


class DecayingPrivacyAllocation(AdaptivePrivacyMechanism):
    """
    Decaying privacy allocation - more privacy early, less later
    Useful when early iterations have more influence on final model
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        decay_rate: float = 0.95,
        sensitivity: float = 1.0,
    ):
        super().__init__(epsilon_budget, delta_budget, total_steps)
        self.decay_rate = decay_rate
        self.sensitivity = sensitivity
        
        # Compute normalization factor for budget allocation
        self.norm_factor = sum(decay_rate ** t for t in range(total_steps))
    
    def get_noise_multiplier(self, step: int) -> float:
        # More budget early on (higher epsilon = less noise)
        budget_fraction = (self.decay_rate ** step) / self.norm_factor
        epsilon_step = self.epsilon_budget * budget_fraction
        delta_step = self.delta_budget * budget_fraction
        
        # Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / delta_step)) * self.sensitivity / epsilon_step
        return sigma
    
    def update_privacy(self, step: int, sensitivity: float):
        budget_fraction = (self.decay_rate ** step) / self.norm_factor
        epsilon_step = self.epsilon_budget * budget_fraction
        delta_step = self.delta_budget * budget_fraction
        
        self.privacy_state = self.privacy_state.update(
            (epsilon_step, delta_step),
            step
        )


class IncreasingPrivacyAllocation(AdaptivePrivacyMechanism):
    """
    Increasing privacy allocation - less privacy early, more later
    Useful when later iterations refine the model more
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        growth_rate: float = 1.05,
        sensitivity: float = 1.0,
    ):
        super().__init__(epsilon_budget, delta_budget, total_steps)
        self.growth_rate = growth_rate
        self.sensitivity = sensitivity
        
        # Compute normalization factor
        self.norm_factor = sum(growth_rate ** t for t in range(total_steps))
    
    def get_noise_multiplier(self, step: int) -> float:
        # More budget later (higher epsilon = less noise)
        budget_fraction = (self.growth_rate ** step) / self.norm_factor
        epsilon_step = self.epsilon_budget * budget_fraction
        delta_step = self.delta_budget * budget_fraction
        
        sigma = np.sqrt(2 * np.log(1.25 / delta_step)) * self.sensitivity / epsilon_step
        return sigma
    
    def update_privacy(self, step: int, sensitivity: float):
        budget_fraction = (self.growth_rate ** step) / self.norm_factor
        epsilon_step = self.epsilon_budget * budget_fraction
        delta_step = self.delta_budget * budget_fraction
        
        self.privacy_state = self.privacy_state.update(
            (epsilon_step, delta_step),
            step
        )


class LossAdaptivePrivacy(AdaptivePrivacyMechanism):
    """
    Loss-adaptive privacy allocation
    Allocates more budget when loss is high (model uncertain)
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        sensitivity: float = 1.0,
        smoothing: float = 0.9,
    ):
        super().__init__(epsilon_budget, delta_budget, total_steps)
        self.sensitivity = sensitivity
        self.smoothing = smoothing  # EMA smoothing for loss
        
        self.loss_history = []
        self.ema_loss = None
        self.remaining_budget = epsilon_budget
        self.remaining_delta = delta_budget
        self.remaining_steps = total_steps
    
    def update_loss(self, loss: float):
        """Update loss history for adaptive allocation"""
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = self.smoothing * self.ema_loss + (1 - self.smoothing) * loss
        
        self.loss_history.append(loss)
    
    def get_noise_multiplier(self, step: int) -> float:
        # If no loss history, use uniform allocation
        if self.ema_loss is None or self.remaining_steps == 0:
            return self._uniform_noise()
        
        # Allocate based on loss magnitude
        # Higher loss = more budget = less noise
        max_expected_loss = 10.0  # Adjust based on task
        loss_weight = min(self.ema_loss / max_expected_loss, 1.0)
        
        # Adaptive budget for this step
        base_epsilon = self.remaining_budget / self.remaining_steps
        epsilon_step = base_epsilon * (1 + loss_weight)
        
        # Ensure we don't exceed budget
        epsilon_step = min(epsilon_step, self.remaining_budget)
        delta_step = (epsilon_step / self.epsilon_budget) * self.delta_budget
        
        sigma = np.sqrt(2 * np.log(1.25 / delta_step)) * self.sensitivity / epsilon_step
        return sigma
    
    def _uniform_noise(self) -> float:
        epsilon_step = self.remaining_budget / max(self.remaining_steps, 1)
        delta_step = self.remaining_delta / max(self.remaining_steps, 1)
        return np.sqrt(2 * np.log(1.25 / delta_step)) * self.sensitivity / epsilon_step
    
    def update_privacy(self, step: int, sensitivity: float, loss: Optional[float] = None):
        if loss is not None:
            self.update_loss(loss)
        
        # Update based on actual noise used
        sigma = self.get_noise_multiplier(step)
        
        # Reverse engineer epsilon from sigma
        epsilon_step = np.sqrt(2 * np.log(1.25 / self.delta_budget)) * sensitivity / sigma
        delta_step = self.delta_budget / self.total_steps
        
        self.remaining_budget -= epsilon_step
        self.remaining_delta -= delta_step
        self.remaining_steps -= 1
        
        self.privacy_state = self.privacy_state.update(
            (epsilon_step, delta_step),
            step
        )


class GradientNormAdaptivePrivacy(AdaptivePrivacyMechanism):
    """
    Gradient norm-adaptive privacy allocation
    Allocates budget based on gradient magnitude
    """
    
    def __init__(
        self,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        clip_norm: float = 1.0,
        smoothing: float = 0.9,
    ):
        super().__init__(epsilon_budget, delta_budget, total_steps)
        self.clip_norm = clip_norm
        self.smoothing = smoothing
        
        self.grad_norm_history = []
        self.ema_grad_norm = None
        self.remaining_budget = epsilon_budget
        self.remaining_delta = delta_budget
        self.remaining_steps = total_steps
    
    def update_grad_norm(self, grad_norm: float):
        """Update gradient norm history"""
        if self.ema_grad_norm is None:
            self.ema_grad_norm = grad_norm
        else:
            self.ema_grad_norm = self.smoothing * self.ema_grad_norm + (1 - self.smoothing) * grad_norm
        
        self.grad_norm_history.append(grad_norm)
    
    def get_noise_multiplier(self, step: int, grad_norm: Optional[float] = None) -> float:
        if grad_norm is not None:
            self.update_grad_norm(grad_norm)
        
        if self.ema_grad_norm is None or self.remaining_steps == 0:
            return self._uniform_noise()
        
        # Large gradients = more uncertainty = more budget needed
        norm_ratio = self.ema_grad_norm / self.clip_norm
        budget_multiplier = 1 + min(norm_ratio, 2.0)
        
        base_epsilon = self.remaining_budget / self.remaining_steps
        epsilon_step = base_epsilon * budget_multiplier
        epsilon_step = min(epsilon_step, self.remaining_budget)
        
        delta_step = (epsilon_step / self.epsilon_budget) * self.delta_budget
        
        sigma = np.sqrt(2 * np.log(1.25 / delta_step)) * self.clip_norm / epsilon_step
        return sigma
    
    def _uniform_noise(self) -> float:
        epsilon_step = self.remaining_budget / max(self.remaining_steps, 1)
        delta_step = self.remaining_delta / max(self.remaining_steps, 1)
        return np.sqrt(2 * np.log(1.25 / delta_step)) * self.clip_norm / epsilon_step
    
    def update_privacy(self, step: int, sensitivity: float, grad_norm: Optional[float] = None):
        if grad_norm is not None:
            self.update_grad_norm(grad_norm)
        
        sigma = self.get_noise_multiplier(step, grad_norm)
        
        epsilon_step = np.sqrt(2 * np.log(1.25 / self.delta_budget)) * sensitivity / sigma
        delta_step = self.delta_budget / self.total_steps
        
        self.remaining_budget -= epsilon_step
        self.remaining_delta -= delta_step
        self.remaining_steps -= 1
        
        self.privacy_state = self.privacy_state.update(
            (epsilon_step, delta_step),
            step
        )


class AdaptivePrivacyFactory:
    """Factory for creating adaptive privacy mechanisms"""
    
    @staticmethod
    def create(
        mechanism_type: str,
        epsilon_budget: float,
        delta_budget: float,
        total_steps: int,
        **kwargs
    ) -> AdaptivePrivacyMechanism:
        """
        Create adaptive privacy mechanism
        
        Args:
            mechanism_type: One of 'uniform', 'decaying', 'increasing', 'loss_adaptive', 'grad_norm_adaptive'
            epsilon_budget: Total privacy budget (epsilon)
            delta_budget: Total privacy budget (delta)
            total_steps: Total number of training steps
            **kwargs: Additional mechanism-specific arguments
        
        Returns:
            AdaptivePrivacyMechanism instance
        """
        if mechanism_type == "uniform":
            return UniformPrivacyAllocation(
                epsilon_budget, delta_budget, total_steps,
                sensitivity=kwargs.get('sensitivity', 1.0)
            )
        elif mechanism_type == "decaying":
            return DecayingPrivacyAllocation(
                epsilon_budget, delta_budget, total_steps,
                decay_rate=kwargs.get('decay_rate', 0.95),
                sensitivity=kwargs.get('sensitivity', 1.0)
            )
        elif mechanism_type == "increasing":
            return IncreasingPrivacyAllocation(
                epsilon_budget, delta_budget, total_steps,
                growth_rate=kwargs.get('growth_rate', 1.05),
                sensitivity=kwargs.get('sensitivity', 1.0)
            )
        elif mechanism_type == "loss_adaptive":
            return LossAdaptivePrivacy(
                epsilon_budget, delta_budget, total_steps,
                sensitivity=kwargs.get('sensitivity', 1.0),
                smoothing=kwargs.get('smoothing', 0.9)
            )
        elif mechanism_type == "grad_norm_adaptive":
            return GradientNormAdaptivePrivacy(
                epsilon_budget, delta_budget, total_steps,
                clip_norm=kwargs.get('clip_norm', 1.0),
                smoothing=kwargs.get('smoothing', 0.9)
            )
        else:
            raise ValueError(f"Unknown mechanism type: {mechanism_type}")


# Utility functions for privacy accounting
def compute_rdp(noise_multiplier: float, steps: int, orders: np.ndarray) -> np.ndarray:
    """
    Compute Rényi Differential Privacy (RDP) guarantees
    
    Args:
        noise_multiplier: Noise multiplier (sigma)
        steps: Number of steps
        orders: Array of Rényi orders (alpha)
    
    Returns:
        RDP epsilon for each order
    """
    return steps / (2 * noise_multiplier ** 2) * orders


def rdp_to_dp(rdp_eps: np.ndarray, orders: np.ndarray, delta: float) -> float:
    """
    Convert RDP to (epsilon, delta)-DP
    
    Args:
        rdp_eps: RDP epsilon values
        orders: Rényi orders
        delta: Target delta
    
    Returns:
        Epsilon for (epsilon, delta)-DP
    """
    eps_values = rdp_eps - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    return np.min(eps_values)
