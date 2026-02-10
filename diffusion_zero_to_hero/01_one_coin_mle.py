"""
Section 1: One Coin Maximum Likelihood Estimation

This demonstrates the simplest case: estimating a coin's bias from observed flips.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# True coin bias (unknown in practice, known here for demonstration)
true_theta = 0.7

# Simulate coin flips
n_flips = 1000
coin_flips = np.random.binomial(1, true_theta, n_flips)

# MLE estimate: fraction of heads
n_heads = np.sum(coin_flips)
theta_mle = n_heads / n_flips

print(f"True bias (theta): {true_theta}")
print(f"Number of flips: {n_flips}")
print(f"Number of heads: {n_heads}")
print(f"MLE estimate: {theta_mle:.4f}")
print(f"Error: {abs(theta_mle - true_theta):.4f}")

# Visualize convergence as we get more data
theta_estimates = []
sample_sizes = range(10, n_flips + 1, 10)

for n in sample_sizes:
    partial_flips = coin_flips[:n]
    theta_est = np.sum(partial_flips) / n
    theta_estimates.append(theta_est)

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, theta_estimates, label='MLE estimate', linewidth=2)
plt.axhline(y=true_theta, color='r', linestyle='--', label=f'True theta = {true_theta}')
plt.xlabel('Number of flips', fontsize=12)
plt.ylabel('Estimated theta', fontsize=12)
plt.title('MLE Convergence: Coin Bias Estimation', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('01_one_coin_mle_convergence.png', dpi=150)
print("\nSaved convergence plot to: 01_one_coin_mle_convergence.png")

# Likelihood function visualization
theta_range = np.linspace(0.01, 0.99, 100)
log_likelihoods = []

for theta in theta_range:
    # Log-likelihood: h*log(theta) + (n-h)*log(1-theta)
    log_likelihood = n_heads * np.log(theta) + (n_flips - n_heads) * np.log(1 - theta)
    log_likelihoods.append(log_likelihood)

# Plot likelihood
plt.figure(figsize=(10, 6))
plt.plot(theta_range, log_likelihoods, linewidth=2, label='Log-likelihood')
plt.axvline(x=theta_mle, color='r', linestyle='--', label=f'MLE = {theta_mle:.4f}')
plt.axvline(x=true_theta, color='g', linestyle='--', label=f'True theta = {true_theta}')
plt.xlabel('Theta (coin bias)', fontsize=12)
plt.ylabel('Log-likelihood', fontsize=12)
plt.title('Log-likelihood Function', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('01_one_coin_mle_likelihood.png', dpi=150)
print("Saved likelihood plot to: 01_one_coin_mle_likelihood.png")

print("\n" + "="*60)
print("KEY INSIGHT:")
print("The MLE is simply the fraction of heads observed.")
print("With more data, the estimate converges to the true bias.")
print("The log-likelihood is maximized at theta = h/n.")
print("="*60)
