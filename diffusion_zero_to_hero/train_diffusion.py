"""Simple training script for the diffusion model.

Usage:
    python train_diffusion.py --epochs 10 --batch-size 32
    python train_diffusion.py --generate --n-samples 16
"""

import argparse
import numpy as np
import jax

from diffusion import (
    DiffusionConfig, UNet, DiffusionModel,
    train_diffusion, visualize_samples, visualize_trajectory
)


def main():
    parser = argparse.ArgumentParser(description="Train or sample from diffusion model")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--generate', action='store_true', help='Generate samples after training')
    parser.add_argument('--n-samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load checkpoint (not implemented yet)')
    args = parser.parse_args()

    config = DiffusionConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_timesteps=args.timesteps
    )

    # Train the model
    print("=" * 60)
    print("Training Diffusion Model")
    print("=" * 60)
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Timesteps: {config.num_timesteps}")
    print("=" * 60)

    model, diffusion, key = train_diffusion(config)

    # Generate samples
    if args.generate:
        print("\n" + "=" * 60)
        print("Generating Samples")
        print("=" * 60)

        key, subkey = jax.random.split(key)
        samples, trajectory = diffusion.sample(
            args.n_samples,
            subkey,
            return_trajectory=True
        )

        # Convert to numpy for visualization
        samples_np = np.array(samples)

        print("\nSaving visualizations...")
        visualize_samples(
            samples_np,
            save_path="diffusion_samples.png",
            title=f"Generated Samples (T={config.num_timesteps})"
        )

        visualize_trajectory(
            trajectory,
            save_path="diffusion_trajectory.png"
        )

        print("Done!")


if __name__ == "__main__":
    main()
