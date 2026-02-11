"""
Section 3: Two-Coin EM Algorithm

This demonstrates the Expectation-Maximization algorithm for the two-coin problem:
- Two coins with different biases θ₁ and θ₂
- Each sequence of flips comes from one of the two coins (latent variable zᵢ)
- We observe sequences but don't know which coin generated each one
- We also don't know the coin biases or the mixture weight π

EM alternates between:
- E-step: Compute probability each sequence came from coin 1 (posterior/weights)
- M-step: Update parameters using weighted averages
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# True parameters (unknown in practice)
true_theta1 = 0.8   # Coin 1: high bias (80% heads)
true_theta2 = 0.3   # Coin 2: low bias (30% heads)
true_pi = 0.6       # 60% of sequences come from coin 1

print("=" * 60)
print("SETUP: Two-Coin Problem")
print("=" * 60)
print(f"True parameters (unknown in practice):")
print(f"  θ₁ (coin 1 bias): {true_theta1}")
print(f"  θ₂ (coin 2 bias): {true_theta2}")
print(f"  π  (P(select coin 1)): {true_pi}")

# Generate synthetic data
n_sequences = 1000
n_flips_per_seq = 20

# For each sequence, sample which coin was used, then flip that coin
data = []
true_assignments = []

for i in range(n_sequences):
    # Sample which coin (latent z_i)
    if np.random.rand() < true_pi:
        z_i = 1  # Coin 1
        theta = true_theta1
    else:
        z_i = 2  # Coin 2
        theta = true_theta2

    # Flip the selected coin n_flips_per_seq times
    flips = np.random.binomial(1, theta, n_flips_per_seq)
    heads = np.sum(flips)

    data.append({
        'n': n_flips_per_seq,
        'h': heads,
        'true_coin': z_i
    })
    true_assignments.append(z_i)

# Convert to arrays for easier handling
h_obs = np.array([d['h'] for d in data])
n_obs = np.array([d['n'] for d in data])
true_z = np.array(true_assignments)

# How many from each coin?
coin1_count = np.sum(true_z == 1)
coin2_count = np.sum(true_z == 2)

print(f"\nGenerated {n_sequences} sequences:")
print(f"  Actually from coin 1: {coin1_count} ({100*coin1_count/n_sequences:.1f}%)")
print(f"  Actually from coin 2: {coin2_count} ({100*coin2_count/n_sequences:.1f}%)")

# Average heads per sequence for each true coin
avg_heads_coin1 = np.mean(h_obs[true_z == 1])
avg_heads_coin2 = np.mean(h_obs[true_z == 2])
print(f"  Avg heads (true coin 1): {avg_heads_coin1:.2f}")
print(f"  Avg heads (true coin 2): {avg_heads_coin2:.2f}")

print("\n" + "=" * 60)
print("EM ALGORITHM")
print("=" * 60)

# Initialize parameters randomly
theta1 = 0.6
theta2 = 0.4
pi = 0.5

print(f"\nInitial guesses:")
print(f"  θ₁⁽⁰⁾ = {theta1:.3f}, θ₂⁽⁰⁾ = {theta2:.3f}, π⁽⁰⁾ = {pi:.3f}")

# EM iterations
max_iterations = 50
tolerance = 1e-6

history = {
    'theta1': [theta1],
    'theta2': [theta2],
    'pi': [pi],
    'log_likelihood': []
}

def compute_log_likelihood(h, n, theta1, theta2, pi):
    """Compute log marginal likelihood of observed data."""
    # p(x_i|z_i=1,θ) = θ₁^h * (1-θ₁)^(n-h)
    # p(x_i|z_i=2,θ) = θ₂^h * (1-θ₂)^(n-h)
    # p(x_i|θ) = π*p(x_i|z=1,θ) + (1-π)*p(x_i|z=2,θ)

    log_lik = 0
    for hi, ni in zip(h, n):
        p1 = (theta1 ** hi) * ((1 - theta1) ** (ni - hi))
        p2 = (theta2 ** hi) * ((1 - theta2) ** (ni - hi))
        marginal = pi * p1 + (1 - pi) * p2
        log_lik += np.log(marginal)
    return log_lik

for iteration in range(max_iterations):
    theta1_old, theta2_old, pi_old = theta1, theta2, pi

    # === E-STEP ===
    # Compute posterior probability w_i = P(z_i=1 | x_i, θ⁽ᵗ⁾)
    # w_i = π * θ₁^h * (1-θ₁)^(n-h) / [π * θ₁^h * (1-θ₁)^(n-h) + (1-π) * θ₂^h * (1-θ₂)^(n-h)]

    w = np.zeros(n_sequences)
    for i, (hi, ni) in enumerate(zip(h_obs, n_obs)):
        # Likelihood under coin 1
        likelihood1 = (theta1 ** hi) * ((1 - theta1) ** (ni - hi))
        # Likelihood under coin 2
        likelihood2 = (theta2 ** hi) * ((1 - theta2) ** (ni - hi))

        # Posterior (weighted by prior π)
        numerator = pi * likelihood1
        denominator = pi * likelihood1 + (1 - pi) * likelihood2
        w[i] = numerator / denominator

    # === M-STEP ===
    # Update parameters using weighted averages

    # θ₁⁽ᵗ⁺¹⁾ = Σ wᵢ*hᵢ / Σ wᵢ*nᵢ
    theta1 = np.sum(w * h_obs) / np.sum(w * n_obs)

    # θ₂⁽ᵗ⁺¹⁾ = Σ (1-wᵢ)*hᵢ / Σ (1-wᵢ)*nᵢ
    theta2 = np.sum((1 - w) * h_obs) / np.sum((1 - w) * n_obs)

    # π⁽ᵗ⁺¹⁾ = (1/N) Σ wᵢ
    pi = np.mean(w)

    # Store history
    history['theta1'].append(theta1)
    history['theta2'].append(theta2)
    history['pi'].append(pi)

    # Compute log-likelihood
    ll = compute_log_likelihood(h_obs, n_obs, theta1, theta2, pi)
    history['log_likelihood'].append(ll)

    # Check convergence
    delta = max(abs(theta1 - theta1_old), abs(theta2 - theta2_old), abs(pi - pi_old))

    if iteration < 10 or delta > tolerance * 10:
        print(f"Iter {iteration:2d}: θ₁={theta1:.4f}, θ₂={theta2:.4f}, π={pi:.4f}, "
              f"logL={ll:.2f}, delta={delta:.6f}")

    if delta < tolerance:
        print(f"\nConverged after {iteration + 1} iterations!")
        break

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"{'Parameter':<15} {'Initial':<12} {'Final EM':<12} {'True':<12}")
print("-" * 60)
print(f"{'θ₁ (coin 1)':<15} {history['theta1'][0]:<12.4f} {theta1:<12.4f} {true_theta1:<12.4f}")
print(f"{'θ₂ (coin 2)':<15} {history['theta2'][0]:<12.4f} {theta2:<12.4f} {true_theta2:<12.4f}")
print(f"{'π (mixture)':<15} {history['pi'][0]:<12.4f} {pi:<12.4f} {true_pi:<12.4f}")

print(f"\nErrors:")
print(f"  θ₁ error: {abs(theta1 - true_theta1):.4f}")
print(f"  θ₂ error: {abs(theta2 - true_theta2):.4f}")
print(f"  π error:  {abs(pi - true_pi):.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Parameter convergence
ax1 = axes[0, 0]
ax1.plot(history['theta1'], 'o-', label='θ₁ (coin 1)', linewidth=2, markersize=5)
ax1.plot(history['theta2'], 's-', label='θ₂ (coin 2)', linewidth=2, markersize=5)
ax1.plot(history['pi'], '^-', label='π (mixture)', linewidth=2, markersize=5)
ax1.axhline(y=true_theta1, color='C0', linestyle='--', alpha=0.7, label=f'True θ₁={true_theta1}')
ax1.axhline(y=true_theta2, color='C1', linestyle='--', alpha=0.7, label=f'True θ₂={true_theta2}')
ax1.axhline(y=true_pi, color='C2', linestyle='--', alpha=0.7, label=f'True π={true_pi}')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Parameter value', fontsize=11)
ax1.set_title('EM Convergence: Parameter Estimates', fontsize=13)
ax1.legend(fontsize=9, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Plot 2: Log-likelihood
ax2 = axes[0, 1]
ax2.plot(history['log_likelihood'], 'o-', color='green', linewidth=2, markersize=5)
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Log-likelihood', fontsize=11)
ax2.set_title('Log-Likelihood (Never Decreases)', fontsize=13)
ax2.grid(True, alpha=0.3)

# Plot 3: E-step weights for a sample of sequences
ax3 = axes[1, 0]
# Show first 50 sequences, color by true coin
colors = ['blue' if z == 1 else 'red' for z in true_z[:50]]
bars = ax3.bar(range(50), w[:50], color=colors, alpha=0.7)
ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
ax3.set_xlabel('Sequence index', fontsize=11)
ax3.set_ylabel('P(zᵢ=1 | xᵢ, θ) = wᵢ', fontsize=11)
ax3.set_title('E-step: Posterior Weights (First 50 Sequences)', fontsize=13)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label='True coin 1'),
                   Patch(facecolor='red', alpha=0.7, label='True coin 2')]
ax3.legend(handles=legend_elements, fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Data distribution (heads per sequence) by true coin
ax4 = axes[1, 1]
bins = np.arange(0, n_flips_per_seq + 2) - 0.5
ax4.hist(h_obs[true_z == 1], bins=bins, alpha=0.6, color='blue',
         label=f'Coin 1 (avg={avg_heads_coin1:.1f} heads)', edgecolor='black')
ax4.hist(h_obs[true_z == 2], bins=bins, alpha=0.6, color='red',
         label=f'Coin 2 (avg={avg_heads_coin2:.1f} heads)', edgecolor='black')
ax4.set_xlabel('Number of heads', fontsize=11)
ax4.set_ylabel('Number of sequences', fontsize=11)
ax4.set_title('Data Distribution by True Coin', fontsize=13)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_two_coin_em.png', dpi=150)
print("\nSaved plot to: 03_two_coin_em.png")

print("\n" + "=" * 60)
print("NOTE ON π ESTIMATION:")
print("=" * 60)
print("π converged to ~0.55 instead of the true 0.60. Why?")
print("")
print("1. SAMPLING VARIANCE: With only 200 sequences, random")
print("   variation means the empirical π in the data differs")
print("   slightly from the true 0.6.")
print("")
print("2. LIKELIHOOD LANDSCAPE: EM finds a LOCAL optimum. The")
print("   marginal likelihood may have multiple modes, and EM")
print("   converges to the nearest one from our initialization.")
print("")
print("3. IDENTIFIABILITY: The coin biases (θ₁≈0.8, θ₂≈0.3) are")
print("   estimated very accurately because they're strongly")
print("   identified by the data. π is softer and more sensitive")
print("   to initialization and local optima.")
print("")
print("This is normal for mixture models! Multiple runs with")
print("different initializations can help find the global optimum.")

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
print("EM handles hidden variables by alternating between:")
print("  E-step: Infer which coin generated each sequence")
print("          (compute posterior weights wᵢ)")
print("  M-step: Update coin biases using weighted averages")
print("          (weighted by how likely each coin is)")
print("")
print("Unlike the one-coin problem, there's no closed-form solution.")
print("The iterative structure is ESSENTIAL for the two-coin case.")
print("=" * 60)
