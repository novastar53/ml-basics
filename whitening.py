import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate correlated data (ellipse shape)
np.random.seed(0)
mean = [0, 0]
cov = [[5, .1], [.1, 5]]  # covariance matrix with off-diagonal terms (correlated)
print("Covariance matrix:")
for row in cov:
    print(row)

X = np.random.multivariate_normal(mean, cov, size=500)

# Step 2: Compute covariance and eigendecomposition
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.cov(X_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 3: Rotate data using eigenvectors
X_rotated = X_centered @ eigenvectors

# Check covariance after rotation
cov_rotated = np.cov(X_rotated, rowvar=False)
print("Covariance after rotation:\n", cov_rotated.round(4))

# Step 4: Whitening (rotating and scaling)
X_whitened = X_rotated / np.sqrt(eigenvalues)

# Check covariance after whitening
cov_whitened = np.cov(X_whitened, rowvar=False)
print("\nCovariance after whitening:\n", cov_whitened.round(4))

# --- PLOTTING SECTION ---
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot original correlated data
axs[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
axs[0].set_title('Original Correlated Data')
axs[0].set_aspect('equal', adjustable='box')

# Plot rotated data
axs[1].scatter(X_rotated[:, 0], X_rotated[:, 1], alpha=0.5, color='g')
axs[1].set_title('Rotated (Decorrelated) Data')
axs[1].set_aspect('equal', adjustable='box')

# Plot whitened data
axs[2].scatter(X_whitened[:, 0], X_whitened[:, 1], alpha=0.5, color='r')
axs[2].set_title('Whitened Data')
axs[2].set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()