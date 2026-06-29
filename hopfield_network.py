import numpy as np
import matplotlib.pyplot as plt


GRID_SIZE = 5
N_NEURONS = GRID_SIZE * GRID_SIZE
NOISE_FRACTION = 0.25
MAX_STEPS = 1000
PATIENCE = 10 * N_NEURONS
RANDOM_SEED = 42
OUTPUT_PNG = "hopfield_recall_grid.png"


def make_letter_patterns():
    """Return a few hand-designed 5x5 bipolar letter patterns."""
    t = np.array(
        [
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
        ]
    ).flatten()

    x = np.array(
        [
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1],
        ]
    ).flatten()

    o = np.array(
        [
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1],
        ]
    ).flatten()

    return {"T": t, "X": x, "O": o}


class HopfieldNetwork:
    """Classical binary Hopfield network with asynchronous updates."""

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """Learn patterns using the Hebbian outer-product rule."""
        for p in patterns:
            self.W += np.outer(p, p)
        self.W /= self.n_neurons
        # No self-connections: a neuron must not feed back into itself.
        np.fill_diagonal(self.W, 0)

    def update(self, state):
        """Perform one asynchronous update on a randomly chosen neuron."""
        i = np.random.randint(self.n_neurons)
        h = self.W[i] @ state
        # Use >= 0 so that h == 0 maps to +1 consistently.
        state[i] = 1 if h >= 0 else -1
        return state

    def energy(self, state):
        """Compute the Hopfield energy E = -0.5 * s^T W s."""
        return -0.5 * state @ self.W @ state

    def recall(self, state, max_steps=MAX_STEPS, patience=PATIENCE):
        """Run asynchronous updates until convergence or max_steps.

        Returns a tuple (snapshots, energies). `snapshots` is a sparse list of
        (step, state, energy) tuples for visualization. `energies` records the
        energy at every update step so the energy trace is smooth.
        """
        state = state.copy()
        snapshots = [(0, state.copy(), self.energy(state))]
        energies = [snapshots[0][2]]
        no_change_count = 0

        for step in range(1, max_steps + 1):
            prev_state = state.copy()
            self.update(state)
            new_energy = self.energy(state)
            energies.append(new_energy)

            if np.array_equal(state, prev_state):
                no_change_count += 1
            else:
                no_change_count = 0

            # Record snapshots at a few fixed steps for the visualization.
            if step in {5, 10, 20}:
                snapshots.append((step, state.copy(), new_energy))

            # Early stopping: if no bit has flipped for `patience` consecutive
            # random updates, the state is almost surely stable.
            if no_change_count >= patience:
                snapshots.append((step, state.copy(), new_energy))
                break

        # Ensure the final state is captured if the loop ended without
        # hitting a snapshot or the convergence branch.
        if snapshots[-1][0] != max_steps and snapshots[-1][0] != step:
            snapshots.append((step, state.copy(), self.energy(state)))

        return snapshots, energies


def corrupt_pattern(pattern, noise_fraction):
    """Flip a fraction of bits in a pattern."""
    noisy = pattern.copy()
    n_flip = int(round(len(pattern) * noise_fraction))
    flip_idx = np.random.choice(len(pattern), size=n_flip, replace=False)
    noisy[flip_idx] *= -1
    return noisy


def build_recall_figure(results):
    """Plot a grid: one row per pattern, columns show recall progression."""
    n_patterns = len(results)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(n_patterns + 1, 6, hspace=0.35, wspace=0.15)

    column_labels = ["Original", "Corrupted", "Step 5", "Step 10", "Step 20", "Converged"]

    for row, (name, snapshots, _) in enumerate(results):
        for col, (title, state) in enumerate(snapshots):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(state.reshape(GRID_SIZE, GRID_SIZE), cmap="binary", vmin=-1, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(name, rotation=0, fontsize=12, labelpad=20, va="center")
            ax.axis("off")

    # Energy trace across all recall runs.
    ax_energy = fig.add_subplot(gs[n_patterns, :])
    for name, _, energies in results:
        ax_energy.plot(energies, label=name, linewidth=1.5)
    ax_energy.set_xlabel("Step", fontsize=11)
    ax_energy.set_ylabel("Energy", fontsize=11)
    ax_energy.set_title("Energy vs. Step", fontsize=12)
    ax_energy.legend(loc="upper right")
    ax_energy.grid(True, alpha=0.3)

    plt.suptitle("Hopfield Network Pattern Recall", fontsize=14, y=0.98)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    np.random.seed(RANDOM_SEED)

    patterns = make_letter_patterns()
    names = list(patterns.keys())
    pattern_list = [patterns[name] for name in names]

    print(f"Stored patterns: {names}")
    print(f"Neurons: {N_NEURONS}, Noise fraction: {NOISE_FRACTION}")

    net = HopfieldNetwork(N_NEURONS)
    net.train(pattern_list)

    results = []
    for name, pattern in patterns.items():
        corrupted = corrupt_pattern(pattern, NOISE_FRACTION)
        snapshots, energies = net.recall(corrupted)

        # Prepare the six snapshots shown in the grid.
        snapshot_lookup = {step: state for step, state, _ in snapshots}
        final_step, final_state, final_energy = snapshots[-1]

        grid_snapshots = [
            ("Original", pattern),
            ("Corrupted", corrupted),
        ]
        for step in (5, 10, 20):
            # If convergence happened earlier, show the converged state.
            use_step = step if step <= final_step else final_step
            grid_snapshots.append((f"Step {step}", snapshot_lookup.get(use_step, final_state)))
        grid_snapshots.append((f"Converged\n(step {final_step})", final_state))

        results.append((name, grid_snapshots, energies))

        matched = np.array_equal(final_state, pattern)
        print(
            f"Pattern {name}: converged at step {final_step}, "
            f"final energy = {final_energy:.4f}, matched original = {matched}"
        )

    build_recall_figure(results)
    print(f"Saved recall grid to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
