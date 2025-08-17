"""
Single Index Model Experiment:
Comparing Baseline SGD vs Edge of Stability Central Flow approaches
"""

import jax
import jax.numpy as jnp
from jax import grad, hessian
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 12

def run_single_index_experiment_same_init():
    # ═══════════════════════════════════════════════════════════
    # PARAMETERS
    # ═══════════════════════════════════════════════════════════
    
    d = 20                    # dimension
    n_samples = 20_000_000     # Monte Carlo pool size
    eta = 0.001               # learning rate
    steps = 500               # iterations
    k_star = 3                # information exponent
    c_kstar = 1.0             # Hermite coefficient
    
    # Random seeds
    key = jax.random.PRNGKey(42)
    key, k1, k2, k_init = jax.random.split(key, 4)
    
    # ═══════════════════════════════════════════════════════════
    # DATA SETUP
    # ═══════════════════════════════════════════════════════════
    
    # Ground truth vector
    w_star = jax.random.normal(k1, (d,))
    w_star /= jnp.linalg.norm(w_star)
    
    # Data pool
    X_pool = jax.random.normal(k2, (n_samples, d))
    y_pool = jnp.maximum(X_pool @ w_star, 0.)**3  # ReLU³ labels
    
    # Loss function and derivatives
    relu3 = lambda z: jnp.maximum(z, 0.)**3
    loss = lambda w: jnp.mean(1.0 - relu3(X_pool @ w) * y_pool)
    grad_L = grad(loss)
    hess_L = hessian(loss)
    sharp = lambda w: jnp.linalg.eigvalsh(hess_L(w))[-1]
    grad_S = grad(sharp)
    
    # ═══════════════════════════════════════════════════════════
    # HERMITE APPROXIMATION FUNCTIONS
    # ═══════════════════════════════════════════════════════════
    
    def grad_L_hermite(w, alpha):
        """Hermite approximation of loss gradient"""
        sum_term = c_kstar**2 / math.factorial(k_star - 1) * alpha**(k_star - 1)
        return -(np.eye(d) - np.outer(w, w)) @ np.array(w_star) * sum_term
    
    def grad_S_hermite(w, alpha):
        """Hermite approximation of sharpness gradient"""
        if k_star - 3 < 0:
            return np.zeros(d)
        coeff = c_kstar**2 / math.factorial(k_star - 3) * alpha**(k_star - 3)
        return (np.eye(d) - np.outer(w, w)) @ np.array(w_star) * coeff
    
    def projection_perp(grad_S_vec):
        """Orthogonal projection operator"""
        norm_gS = np.linalg.norm(grad_S_vec)
        if norm_gS < 1e-12:
            return np.eye(d)
        return np.eye(d) - np.outer(grad_S_vec, grad_S_vec) / (norm_gS**2)
    
    # Generate initial weight vector for all three methods
    w_init = jax.random.normal(k_init, (d,))
    w_init /= jnp.linalg.norm(w_init)
    
    # Copy this initial weight to all three methods
    w_baseline = w_init.copy()
    w_eos_central = w_init.copy()  
    w_hermite = np.array(w_init)  # Convert to numpy for hermite method
    
    # Compute initial alignment
    initial_alpha = float(w_init @ w_star)
    
    print(f"Shared initial alignment: α₀ = {initial_alpha:+.4f}")
    print(f"Expected random alignment: ~±{1/np.sqrt(d):.4f}")
    
    # Start all arrays with the initial alignment at t=0
    alpha_baseline = [initial_alpha]
    gradnorm_baseline = []  # No gradient at t=0
    
    alpha_eos_central = [initial_alpha]  
    vnorm_eos_central = []  # No update at t=0
    
    alpha_hermite = [initial_alpha]
    vnorm_hermite = []  # No update at t=0
    
    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 1: BASELINE SGD
    # ═══════════════════════════════════════════════════════════
    
    print("\nRunning Baseline SGD...")
    for t in range(steps):
        # Compute gradient
        gL = grad_L(w_baseline)
        
        # Update step
        w_baseline = w_baseline - eta * gL
        w_baseline = w_baseline / jnp.linalg.norm(w_baseline)
        
        # Record metrics
        alpha_baseline.append(float(w_baseline @ w_star))
        gradnorm_baseline.append(float(jnp.linalg.norm(gL)))
        
        if t % 100 == 0:
            print(f"  Step {t:3d}: α = {alpha_baseline[-1]:+.4f}, ||∇L|| = {gradnorm_baseline[-1]:.3e}")
    
    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 2: EOS CENTRAL FLOW (AUTOMATIC DIFFERENTIATION)
    # ═══════════════════════════════════════════════════════════
    
    print("\nRunning EoS Central Flow...")
    for t in range(steps):
        # Compute gradients
        gL = grad_L(w_eos_central)
        gS = grad_S(w_eos_central)
        
        # EoS central flow update: v_t = -Π⊥_{∇S} ∇L
        denom = jnp.dot(gS, gS) + 1e-12  # numerical stability
        v_t = -(gL - gS * (jnp.dot(gL, gS) / denom))
        
        # Update step
        w_eos_central = w_eos_central + eta * v_t
        w_eos_central = w_eos_central / jnp.linalg.norm(w_eos_central)
        
        # Record metrics
        alpha_eos_central.append(float(w_eos_central @ w_star))
        vnorm_eos_central.append(float(jnp.linalg.norm(v_t)))
        
        if t % 100 == 0:
            print(f"  Step {t:3d}: α = {alpha_eos_central[-1]:+.4f}, ||v_t|| = {vnorm_eos_central[-1]:.3e}")
    
    # ═══════════════════════════════════════════════════════════
    # EXPERIMENT 3: EOS CENTRAL FLOW (HERMITE APPROXIMATION)
    # ═══════════════════════════════════════════════════════════
    
    print("\nRunning EoS Hermite Approximation...")
    for t in range(steps):
        # Current alignment
        alpha = np.dot(w_hermite, np.array(w_star))
        
        # Hermite-based gradients
        gL_hermite = grad_L_hermite(w_hermite, alpha)
        gS_hermite = grad_S_hermite(w_hermite, alpha)
        
        # EoS central flow update with projection
        Π_perp = projection_perp(gS_hermite)
        v_t = -Π_perp @ gL_hermite
        
        # Update step
        w_new = w_hermite + eta * v_t
        w_hermite = w_new / np.linalg.norm(w_new)
        
        # Record metrics
        alpha_hermite.append(float(np.dot(w_hermite, np.array(w_star))))
        vnorm_hermite.append(float(np.linalg.norm(v_t)))
        
        if t % 100 == 0:
            print(f"  Step {t:3d}: α = {alpha_hermite[-1]:+.4f}, ||v_t|| = {vnorm_hermite[-1]:.3e}")
    
    # ═══════════════════════════════════════════════════════════
    # VISUALIZATION - INCLUDING t=0!
    # ═══════════════════════════════════════════════════════════
    
    # Time steps now start from 0 and include initial state
    time_steps_alpha = np.arange(0, steps + 1)  # 0, 1, 2, ..., 500
    time_steps_updates = np.arange(1, steps + 1)  # 1, 2, ..., 500 (no update at t=0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for consistency
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Sea Green, Crimson, Royal Blue
    
    # Plot 1: Alignment α vs Time (starting t=0)
    ax1.plot(time_steps_alpha, alpha_baseline, label='Baseline SGD', 
             color=colors[0], linewidth=2.5, alpha=0.8)
    ax1.plot(time_steps_alpha, alpha_eos_central, label='EoS Central Flow', 
             color=colors[1], linewidth=2.5, alpha=0.8)
    ax1.plot(time_steps_alpha, alpha_hermite, label='EoS Hermite Approx.', 
             color=colors[2], linewidth=2.5, alpha=0.8, linestyle='--')
    
    ax1.set_xlabel('Time Step $t$', fontsize=14)
    ax1.set_ylabel('Alignment $\\alpha = w \\cdot w^*$', fontsize=14)
    ax1.set_title('Alignment Evolution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=11, loc='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlim(-10, steps + 10)  # Add some padding
    
    # Plot 2: Gradient/Update Norms vs Time (starting from t=1)
    ax2.plot(time_steps_updates, gradnorm_baseline, label='Baseline SGD', 
             color=colors[0], linewidth=2.5, alpha=0.8)
    ax2.plot(time_steps_updates, vnorm_eos_central, label='EoS Central Flow', 
             color=colors[1], linewidth=2.5, alpha=0.8)
    ax2.plot(time_steps_updates, vnorm_hermite, label='EoS Hermite Approx.', 
             color=colors[2], linewidth=2.5, alpha=0.8, linestyle='--')
    
    ax2.set_xlabel('Time Step $t$', fontsize=14)
    ax2.set_ylabel('Update Vector Magnitude $||v_t||$', fontsize=14)
    ax2.set_title('Update Vector Evolution', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to show small values clearly
    ax2.set_xlim(0, steps + 10)
    
    plt.tight_layout()
    plt.savefig('single_idx.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS SUMMARY (Same Initial Conditions)")
    print(f"{'='*70}")
    print(f"Shared initial alignment:            α₀ = {initial_alpha:+.4f}")
    print(f"Baseline SGD final alignment:        α = {alpha_baseline[-1]:+.4f}")
    print(f"EoS Central Flow final alignment:    α = {alpha_eos_central[-1]:+.4f}")
    print(f"EoS Hermite Approx final alignment:  α = {alpha_hermite[-1]:+.4f}")
    print(f"\nAlignment improvements from initial:")
    print(f"Baseline SGD: Δα = {alpha_baseline[-1] - initial_alpha:+.4f}")
    print(f"EoS Central Flow: Δα = {alpha_eos_central[-1] - initial_alpha:+.4f}")
    print(f"EoS Hermite: Δα = {alpha_hermite[-1] - initial_alpha:+.4f}")
    print(f"\nFinal update magnitudes:")
    print(f"Baseline SGD: ||∇L|| = {gradnorm_baseline[-1]:.3e}")
    print(f"EoS Central Flow: ||v_t|| = {vnorm_eos_central[-1]:.3e}")
    print(f"EoS Hermite: ||v_t|| = {vnorm_hermite[-1]:.3e}")
    
    return {
        'initial_alpha': initial_alpha,
        'alpha_baseline': alpha_baseline,
        'alpha_eos_central': alpha_eos_central,
        'alpha_hermite': alpha_hermite,
        'gradnorm_baseline': gradnorm_baseline,
        'vnorm_eos_central': vnorm_eos_central,
        'vnorm_hermite': vnorm_hermite
    }

if __name__ == "__main__":
    results = run_single_index_experiment_same_init()
