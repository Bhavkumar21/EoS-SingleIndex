import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-ready style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# Color palette
colors = sns.color_palette("husl", 2)
sgd_color = colors[0]  # Blue-ish
eos_color = colors[1]  # Red-ish

# Activation and derivatives
def relu3(z):
    return np.maximum(z, 0)**3

def relu3_prime(z):
    return 3 * np.maximum(z, 0)**2

def relu3_double_prime(z):
    return 6 * np.maximum(z, 0)

# Compute loss
def compute_loss(w, X, y):
    return np.mean(1 - relu3(X @ w) * y)

# Analytical gradient of loss
def compute_grad_loss(w, X, y):
    z = X @ w
    return -np.mean(X * (relu3_prime(z) * y)[:, np.newaxis], axis=0)

# Analytical Hessian of loss
def compute_hessian(w, X, y):
    z = X @ w
    weights = relu3_double_prime(z) * y
    H = np.zeros((len(w), len(w)))
    for i in range(len(X)):
        H -= np.outer(X[i], X[i]) * weights[i]
    return H / len(X)

# Gradient of sharpness (numerical, but on max eigenvalue)
def compute_grad_sharpness(w, X, y, eps=1e-6):
    def max_eig(w_):
        H = compute_hessian(w_, X, y)
        evals = np.linalg.eigvalsh(H)
        return evals[-1] if np.all(np.isfinite(evals)) else np.nan
    
    grad_S = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_m = w.copy()
        w_p[i] += eps
        w_m[i] -= eps
        val_p = max_eig(w_p)
        val_m = max_eig(w_m)
        if np.isfinite(val_p) and np.isfinite(val_m):
            grad_S[i] = (val_p - val_m) / (2 * eps)
        else:
            grad_S[i] = 0  # Handle nan
    return grad_S

def run_experiment():
    # Parameters
    d = 10
    n = 10000
    eta = 0.001
    steps = 1000
    
    np.random.seed(42)
    
    # Data
    w_star = np.random.randn(d)
    w_star /= np.linalg.norm(w_star)
    
    X = np.random.randn(n, d)
    y = relu3(X @ w_star)
    
    # Initial weights
    w_init = np.random.randn(d)
    w_init /= np.linalg.norm(w_init)
    
    w_sgd = w_init.copy()
    w_eos = w_init.copy()
    
    # Storage
    alpha_sgd = [np.dot(w_sgd, w_star)]
    alpha_eos = [np.dot(w_eos, w_star)]
    v_sgd_norms = []
    v_eos_norms = []
    
    print(f"Initial alignment: {alpha_sgd[0]:.6f}")
    
    for step in range(steps):
        # SGD
        grad_L_sgd = compute_grad_loss(w_sgd, X, y)
        v_sgd = -eta * grad_L_sgd
        w_sgd += v_sgd
        w_sgd /= np.linalg.norm(w_sgd)  # normalize to prevent explosion
        
        # EoS
        grad_L_eos = compute_grad_loss(w_eos, X, y)
        grad_S_eos = compute_grad_sharpness(w_eos, X, y)
        
        denom = np.dot(grad_S_eos, grad_S_eos) + 1e-12
        proj = np.eye(d) - np.outer(grad_S_eos, grad_S_eos) / denom
        
        v_eos = -eta * (proj @ grad_L_eos)
        w_eos += v_eos
        w_eos /= np.linalg.norm(w_eos)  # normalize
        
        # Record
        alpha_sgd.append(np.dot(w_sgd, w_star))
        alpha_eos.append(np.dot(w_eos, w_star))
        v_sgd_norms.append(np.linalg.norm(v_sgd))
        v_eos_norms.append(np.linalg.norm(v_eos))
        
        if step % 20 == 0:
            print(f"Step {step:3d}: SGD α={alpha_sgd[-1]:.4f}, EoS α={alpha_eos[-1]:.4f}")
            print(f"           |v_sgd|={v_sgd_norms[-1]:.6f}, |v_eos|={v_eos_norms[-1]:.6f}")
    
    return alpha_sgd, alpha_eos, v_sgd_norms, v_eos_norms

# Run experiment
alpha_sgd, alpha_eos, v_sgd_norms, v_eos_norms = run_experiment()

# Create publication-ready plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Time arrays
time_alpha = np.arange(len(alpha_sgd))
time_v = np.arange(1, len(v_sgd_norms) + 1)

# Left subplot: Alignment
ax1.plot(time_alpha, alpha_sgd, color=sgd_color, linewidth=3, 
         label='SGD Alignment', alpha=0.9)
ax1.plot(time_alpha, alpha_eos, color=eos_color, linewidth=3, 
         label='EoS Alignment', alpha=0.9)

ax1.set_xlabel('Step', fontsize=14, fontweight='bold')
ax1.set_ylabel(r'Alignment $\alpha = w \cdot w^*$', fontsize=14, fontweight='bold')
ax1.set_title('Alignment Across Steps', fontsize=16, fontweight='bold', pad=20)

# Dynamic y-limits for alignment (include negatives with padding)
alpha_min = min(min(alpha_sgd), min(alpha_eos))
alpha_max = max(max(alpha_sgd), max(alpha_eos))
alpha_range = alpha_max - alpha_min
padding = 0.1 * alpha_range if alpha_range > 0 else 0.1
ax1.set_ylim(alpha_min - padding, alpha_max + padding)

ax1.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=12)

# Right subplot: Update norm
ax2.plot(time_v, v_sgd_norms, color=sgd_color, linewidth=3, 
         label=r'SGD $|v_t|$', alpha=0.9)
ax2.plot(time_v, v_eos_norms, color=eos_color, linewidth=3, 
         label=r'EoS $|v_t|$', alpha=0.9)

ax2.set_xlabel('Step', fontsize=14, fontweight='bold')
ax2.set_ylabel(r'Update Norm $|v_t|$', fontsize=14, fontweight='bold')
ax2.set_title('Update Norm Across Steps', fontsize=16, fontweight='bold', pad=20)

ax2.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=12)

# Tight layout and save
plt.tight_layout()
plt.savefig('single_index_eos_sgd_comparison.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

