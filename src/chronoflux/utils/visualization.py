import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(g_losses, d_losses, save_path="training_curves.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.title("Training Curves")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def save_image_grid(images, titles, filename="comparison.png"):
    fig = plt.figure(figsize=(12, 6))
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(1, len(images), i+1)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    plt.savefig(filename)
    plt.close()