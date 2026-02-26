"""Train CLIP-conditioned VAE on CelebA."""
import warnings
warnings.filterwarnings('ignore')

from vae_clip import train_clip_vae, Config, generate_from_text, visualize_generation

# Train for 2 epochs as a test
print("Starting training...")
model, clip_wrapper, key = train_clip_vae(num_epochs=10, batch_size=32, learning_rate=1e-3)

# Generate samples after training
print("\nGenerating samples from trained model...")
prompts = ['a smiling person', 'a person with glasses']
images = generate_from_text(model, clip_wrapper, prompts, key, Config())
visualize_generation(images, titles=prompts, save_path='trained_vae_samples.png')
print('Training complete! Saved samples to trained_vae_samples.png')
