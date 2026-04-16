import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def generate_dataset(n_samples=50, n_clusters=3, cluster_std=1.5, random_state=42):
    data, true_labels = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return data, true_labels


def plot_dataset(data, title="Generated Data"):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.7, edgecolors='k', s=50)
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


class GaussianMixture:
    def __init__(self, n_components=3, max_iter=1000, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means = None
        self.covariances = None
        self.weights = None
        self.converged = False
        self.n_iter = 0

    def _initialize(self, X):
        n_samples, n_features = X.shape
        np.random.seed(self.random_state)

        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[idx].copy()

        self.covariances = np.array([
            np.eye(n_features) for _ in range(self.n_components)
        ])

        self.weights = np.ones(self.n_components) / self.n_components

    def _compute_pdf(self, X, k):
        mean = self.means[k]
        cov = self.covariances[k]
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)
        norm = 1.0 / (np.power(2 * np.pi, X.shape[1] / 2) * np.sqrt(det))
        exp = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm * exp

    def _compute_log_likelihood(self, X):
        log_likelihood = 0.0
        for n in range(X.shape[0]):
            p = 0.0
            for k in range(self.n_components):
                p += self.weights[k] * self._compute_pdf(X[n:n+1], k)[0]
            log_likelihood += np.log(p + 1e-300)
        return log_likelihood

    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._compute_pdf(X, k)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True) + 1e-300
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        N_k = responsibilities.sum(axis=0)

        self.weights = N_k / n_samples

        for k in range(self.n_components):
            if N_k[k] > 1e-10:
                self.means[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / N_k[k]

                diff = X - self.means[k]
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances[k] = (weighted_diff.T @ diff) / N_k[k]

                self.covariances[k] += 1e-6 * np.eye(n_features)

    def fit(self, X):
        self._initialize(X)
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)

            ll = self._compute_log_likelihood(X)
            self.n_iter = iteration + 1

            if abs(ll - prev_ll) < self.tol:
                self.converged = True
                break

            prev_ll = ll

        return self

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        return self._e_step(X)


def plot_clustered_results(data, labels, gmm, true_labels=None):
    plt.figure(figsize=(10, 6))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    for k in range(gmm.n_components):
        mask = labels == k
        plt.scatter(data[mask, 0], data[mask, 1], c=colors[k], alpha=0.7,
                    edgecolors='k', s=60, label=f'Cluster {k}')

    for k in range(gmm.n_components):
        mean = gmm.means[k]
        cov = gmm.covariances[k]
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        
        width, height = 2 * np.sqrt(2 * eigenvalues)
        
        ellipse = plt.matplotlib.patches.Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            fill=False, edgecolor=colors[k], linewidth=2, linestyle='--'
        )
        plt.gca().add_patch(ellipse)

    plt.scatter(gmm.means[:, 0], gmm.means[:, 1], c='black', marker='X',
                s=200, edgecolors='white', linewidth=2, zorder=10, label='Means')

    plt.title(f'GMM Clustering Results (converged in {gmm.n_iter} iterations)')
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    data, true_labels = generate_dataset(n_samples=50, n_clusters=3, cluster_std=1.5, random_state=42)
    
    plot_dataset(data, "Unlabeled Generated Data")
    
    gmm = GaussianMixture(n_components=3, max_iter=200, tol=1e-6, random_state=123)
    gmm.fit(data)
    
    pred_labels = gmm.predict(data)
    
    plot_clustered_results(data, pred_labels, gmm, true_labels)


if __name__ == "__main__":
    main()