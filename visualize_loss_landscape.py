import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_mnist import LinearMNIST

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data(batch_size=128):
    """Load MNIST dataset with same normalization as training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def filter_normalize_direction(direction, weights):
    """
    Apply filter-wise normalization to direction vectors.
    For FC layers, each row (output neuron) is treated as a filter.
    
    Args:
        direction: Random direction tensor with same shape as weights
        weights: Model weights tensor
    
    Returns:
        Normalized direction tensor
    """
    if len(weights.shape) == 2:  # Fully connected layer
        # Each row is a filter (weights for one output neuron)
        normalized = torch.zeros_like(direction)
        for i in range(weights.shape[0]):
            d_norm = torch.norm(direction[i])
            w_norm = torch.norm(weights[i])
            if d_norm > 1e-10:  # Avoid division by zero
                normalized[i] = (direction[i] / d_norm) * w_norm
            else:
                normalized[i] = direction[i]
        return normalized
    else:
        # For other tensor shapes, return as-is
        return direction

def create_random_directions(model, ignore_bias=True):
    """
    Create two random direction vectors with filter-wise normalization.
    
    Args:
        model: The neural network model
        ignore_bias: If True, set bias directions to zero
    
    Returns:
        Two dictionaries containing direction vectors for each parameter
    """
    delta = {}
    eta = {}
    
    for name, param in model.named_parameters():
        if ignore_bias and 'bias' in name:
            # Set bias direction to zero (ignore bias perturbation)
            delta[name] = torch.zeros_like(param)
            eta[name] = torch.zeros_like(param)
        else:
            # Sample from standard normal distribution
            delta[name] = torch.randn_like(param)
            eta[name] = torch.randn_like(param)
            
            # Apply filter-wise normalization
            delta[name] = filter_normalize_direction(delta[name], param)
            eta[name] = filter_normalize_direction(eta[name], param)
    
    return delta, eta

def gram_schmidt(delta, eta):
    """
    Optional: Apply Gram-Schmidt to make directions more orthogonal.
    
    Args:
        delta, eta: Direction dictionaries
    
    Returns:
        Orthogonalized eta
    """
    # Compute inner product
    inner_product = 0
    delta_norm_sq = 0
    
    for name in delta.keys():
        inner_product += torch.sum(delta[name] * eta[name]).item()
        delta_norm_sq += torch.sum(delta[name] * delta[name]).item()
    
    # Orthogonalize eta with respect to delta
    if delta_norm_sq > 1e-10:
        eta_ortho = {}
        for name in eta.keys():
            eta_ortho[name] = eta[name] - (inner_product / delta_norm_sq) * delta[name]
        return eta_ortho
    else:
        return eta

def precompute_logits(model, data_loader, delta, eta, device):
    """
    Precompute logits for efficient loss landscape computation.
    Returns Z_star, Z_delta, Z_eta for all samples.
    
    Args:
        model: The trained model
        data_loader: DataLoader for the dataset
        delta, eta: Direction dictionaries
        device: Computation device
    
    Returns:
        Z_star, Z_delta, Z_eta (all samples' logits), labels
    """
    model.eval()
    
    Z_star_list = []
    Z_delta_list = []
    Z_eta_list = []
    labels_list = []
    
    # Get original weights
    W_star = model.fc.weight.data.clone()
    b_star = model.fc.bias.data.clone()
    
    # Get direction weights (bias is zero if ignored)
    W_delta = delta['fc.weight']
    W_eta = eta['fc.weight']
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            
            # Flatten images: (batch, 1, 28, 28) -> (batch, 784)
            X = data.view(data.size(0), -1)
            
            # Compute logits for each direction
            # Z_star = W_star @ X^T + b_star
            Z_star = torch.mm(X, W_star.t()) + b_star
            
            # Z_delta = W_delta @ X^T (no bias)
            Z_delta = torch.mm(X, W_delta.t())
            
            # Z_eta = W_eta @ X^T (no bias)
            Z_eta = torch.mm(X, W_eta.t())
            
            Z_star_list.append(Z_star.cpu())
            Z_delta_list.append(Z_delta.cpu())
            Z_eta_list.append(Z_eta.cpu())
            labels_list.append(target)
    
    Z_star = torch.cat(Z_star_list, dim=0)
    Z_delta = torch.cat(Z_delta_list, dim=0)
    Z_eta = torch.cat(Z_eta_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return Z_star, Z_delta, Z_eta, labels

def compute_loss_at_point(Z_star, Z_delta, Z_eta, labels, alpha, beta):
    """
    Compute cross-entropy loss at point (alpha, beta) in the 2D slice.
    
    Args:
        Z_star, Z_delta, Z_eta: Precomputed logits
        labels: Ground truth labels
        alpha, beta: Coordinates in the 2D loss landscape
    
    Returns:
        Loss value
    """
    # Compute combined logits: Z = Z_star + alpha * Z_delta + beta * Z_eta
    Z = Z_star + alpha * Z_delta + beta * Z_eta
    
    # Compute cross-entropy loss using log-sum-exp for numerical stability
    # CE = mean(LSE(Z) - Z[range(m), labels])
    log_sum_exp = torch.logsumexp(Z, dim=1)
    correct_class_logits = Z[range(len(labels)), labels]
    loss = torch.mean(log_sum_exp - correct_class_logits)
    
    return loss.item()

def compute_loss_grid(Z_star, Z_delta, Z_eta, labels, alpha_range, beta_range, resolution=51):
    """
    Compute loss values over a 2D grid.
    
    Args:
        Z_star, Z_delta, Z_eta: Precomputed logits
        labels: Ground truth labels
        alpha_range: (min, max) for alpha axis
        beta_range: (min, max) for beta axis
        resolution: Number of points along each axis
    
    Returns:
        alpha_grid, beta_grid, loss_grid
    """
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], resolution)
    beta_vals = np.linspace(beta_range[0], beta_range[1], resolution)
    
    loss_grid = np.zeros((resolution, resolution))
    
    print(f"Computing {resolution}x{resolution} loss grid...")
    for i, alpha in enumerate(alpha_vals):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{resolution} rows completed")
        for j, beta in enumerate(beta_vals):
            loss_grid[i, j] = compute_loss_at_point(
                Z_star, Z_delta, Z_eta, labels, alpha, beta
            )
    
    return alpha_vals, beta_vals, loss_grid

def plot_2d_contour(alpha_vals, beta_vals, loss_grid, title, save_path, 
                    vmin=None, vmax=None):
    """
    Plot 2D contour map of the loss landscape.
    
    Args:
        alpha_vals, beta_vals: Grid coordinates
        loss_grid: Loss values
        title: Plot title
        save_path: Path to save the figure
        vmin, vmax: Value range for colormap
    """
    plt.figure(figsize=(10, 8))
    
    # Create meshgrid for contour plot
    Alpha, Beta = np.meshgrid(alpha_vals, beta_vals)
    
    # Plot filled contours
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 30)
        contourf = plt.contourf(Alpha, Beta, loss_grid.T, levels=levels, cmap='viridis')
    else:
        contourf = plt.contourf(Alpha, Beta, loss_grid.T, levels=30, cmap='viridis')
    
    # Add contour lines
    contour = plt.contour(Alpha, Beta, loss_grid.T, levels=15, colors='white', 
                          alpha=0.4, linewidths=0.5)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Mark the center point (0, 0) - the trained model
    plt.plot(0, 0, marker='*', color='red', markersize=20, 
             markeredgecolor='white', markeredgewidth=1.5, label='Trained model (θ*)')
    
    # Labels and formatting
    plt.xlabel('α direction (filter-normalized)', fontsize=12)
    plt.ylabel('β direction (filter-normalized)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar(contourf, label='Loss value')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_3d_surface(alpha_vals, beta_vals, loss_grid, title, save_path):
    """
    Plot 3D surface of the loss landscape.
    
    Args:
        alpha_vals, beta_vals: Grid coordinates
        loss_grid: Loss values
        title: Plot title
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    Alpha, Beta = np.meshgrid(alpha_vals, beta_vals)
    
    # Plot surface
    surf = ax.plot_surface(Alpha, Beta, loss_grid.T, cmap='viridis', 
                          alpha=0.8, edgecolor='none')
    
    # Mark the center point
    center_loss = loss_grid[len(alpha_vals)//2, len(beta_vals)//2]
    ax.scatter([0], [0], [center_loss], color='red', s=100, marker='*', 
               label='Trained model (θ*)')
    
    # Labels
    ax.set_xlabel('α direction', fontsize=11)
    ax.set_ylabel('β direction', fontsize=11)
    ax.set_zlabel('Loss value', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.legend()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_1d_slice(alpha_vals, beta_vals, loss_grid, title, save_path):
    """
    Plot 1D slices along alpha=0 and beta=0.
    
    Args:
        alpha_vals, beta_vals: Grid coordinates
        loss_grid: Loss values
        title: Plot title
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Find center indices
    center_alpha = len(alpha_vals) // 2
    center_beta = len(beta_vals) // 2
    
    # Plot slice along alpha (beta=0)
    ax1.plot(alpha_vals, loss_grid[:, center_beta], 'b-', linewidth=2)
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='θ*')
    ax1.set_xlabel('α (β=0)', fontsize=12)
    ax1.set_ylabel('Loss value', fontsize=12)
    ax1.set_title('1D slice along α direction', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot slice along beta (alpha=0)
    ax2.plot(beta_vals, loss_grid[center_alpha, :], 'g-', linewidth=2)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='θ*')
    ax2.set_xlabel('β (α=0)', fontsize=12)
    ax2.set_ylabel('Loss value', fontsize=12)
    ax2.set_title('1D slice along β direction', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    # Configuration
    SEED = 42
    RESOLUTION = 51  # Default resolution as per paper
    ALPHA_RANGE = (-1, 1)
    BETA_RANGE = (-1, 1)
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("=" * 60)
    print("MNIST Linear Classifier - Loss Landscape Visualization")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Random seed: {SEED}")
    print(f"Grid resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Alpha range: {ALPHA_RANGE}")
    print(f"Beta range: {BETA_RANGE}")
    print("=" * 60)
    
    # Set random seed
    set_seed(SEED)
    
    # Create output directory
    os.makedirs('loss_landscape_results', exist_ok=True)
    
    # Load model
    print("\n1. Loading trained model...")
    model = LinearMNIST().to(DEVICE)
    checkpoint = torch.load('mnist_linear_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Model loaded. Test accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    # Load data
    print("\n2. Loading MNIST dataset...")
    train_loader, test_loader = load_data(batch_size=512)
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Create random directions with filter-wise normalization
    print("\n3. Creating filter-normalized random directions...")
    delta, eta = create_random_directions(model, ignore_bias=True)
    
    # Optional: Apply Gram-Schmidt
    print("   Applying Gram-Schmidt orthogonalization...")
    eta = gram_schmidt(delta, eta)
    print("   Directions created and normalized.")
    
    # Precompute logits for training set
    print("\n4. Precomputing logits for training set...")
    Z_star_train, Z_delta_train, Z_eta_train, labels_train = precompute_logits(
        model, train_loader, delta, eta, DEVICE
    )
    print(f"   Logits computed: {Z_star_train.shape}")
    
    # Verify center point loss
    center_loss_train = compute_loss_at_point(
        Z_star_train, Z_delta_train, Z_eta_train, labels_train, 0, 0
    )
    print(f"   Center point (0,0) loss: {center_loss_train:.6f}")
    
    # Compute loss grid for training set
    print("\n5. Computing loss landscape grid for training set...")
    alpha_vals, beta_vals, loss_grid_train = compute_loss_grid(
        Z_star_train, Z_delta_train, Z_eta_train, labels_train,
        ALPHA_RANGE, BETA_RANGE, RESOLUTION
    )
    print(f"   Loss grid computed. Min: {loss_grid_train.min():.4f}, Max: {loss_grid_train.max():.4f}")
    
    # Save numerical data
    print("\n6. Saving numerical data...")
    np.savez('loss_landscape_results/loss_grid_train.npz',
             alpha=alpha_vals,
             beta=beta_vals,
             loss=loss_grid_train,
             center_loss=center_loss_train,
             resolution=RESOLUTION,
             seed=SEED)
    print("   Saved: loss_landscape_results/loss_grid_train.npz")
    
    # Plot 2D contour for training set
    print("\n7. Creating visualizations...")
    plot_2d_contour(
        alpha_vals, beta_vals, loss_grid_train,
        'MNIST Linear Classifier - Training Loss Landscape\n(Filter-Normalized Random Directions)',
        'loss_landscape_results/landscape_train_2d_contour.png'
    )
    
    # Plot 3D surface
    plot_3d_surface(
        alpha_vals, beta_vals, loss_grid_train,
        'MNIST Linear Classifier - Training Loss Landscape (3D)',
        'loss_landscape_results/landscape_train_3d_surface.png'
    )
    
    # Plot 1D slices
    plot_1d_slice(
        alpha_vals, beta_vals, loss_grid_train,
        'MNIST Linear Classifier - 1D Loss Slices',
        'loss_landscape_results/landscape_train_1d_slices.png'
    )
    
    # Optional: Compute for test set
    print("\n8. Computing loss landscape for test set (optional)...")
    Z_star_test, Z_delta_test, Z_eta_test, labels_test = precompute_logits(
        model, test_loader, delta, eta, DEVICE
    )
    
    center_loss_test = compute_loss_at_point(
        Z_star_test, Z_delta_test, Z_eta_test, labels_test, 0, 0
    )
    print(f"   Test set center point (0,0) loss: {center_loss_test:.6f}")
    
    alpha_vals_test, beta_vals_test, loss_grid_test = compute_loss_grid(
        Z_star_test, Z_delta_test, Z_eta_test, labels_test,
        ALPHA_RANGE, BETA_RANGE, RESOLUTION
    )
    
    # Save test data
    np.savez('loss_landscape_results/loss_grid_test.npz',
             alpha=alpha_vals_test,
             beta=beta_vals_test,
             loss=loss_grid_test,
             center_loss=center_loss_test,
             resolution=RESOLUTION,
             seed=SEED)
    
    # Plot test set contour
    plot_2d_contour(
        alpha_vals_test, beta_vals_test, loss_grid_test,
        'MNIST Linear Classifier - Test Loss Landscape\n(Filter-Normalized Random Directions)',
        'loss_landscape_results/landscape_test_2d_contour.png'
    )
    
    print("\n" + "=" * 60)
    print("Loss landscape analysis completed!")
    print("Results saved in: loss_landscape_results/")
    print("=" * 60)
    
    # Generate experiment log
    with open('loss_landscape_results/experiment_log.txt', 'w') as f:
        f.write("MNIST Linear Classifier - Loss Landscape Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Random seed: {SEED}\n")
        f.write(f"Grid resolution: {RESOLUTION}x{RESOLUTION}\n")
        f.write(f"Alpha range: {ALPHA_RANGE}\n")
        f.write(f"Beta range: {BETA_RANGE}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Normalization method: Filter-wise\n")
        f.write(f"Bias handling: Ignored (set to zero)\n\n")
        f.write(f"Training set center loss L(0,0): {center_loss_train:.6f}\n")
        f.write(f"Test set center loss L(0,0): {center_loss_test:.6f}\n")
        f.write(f"Model test accuracy: {checkpoint['test_accuracy']:.2f}%\n\n")
        f.write("Files generated:\n")
        f.write("- loss_grid_train.npz (numerical data)\n")
        f.write("- loss_grid_test.npz (numerical data)\n")
        f.write("- landscape_train_2d_contour.png (2D contour plot)\n")
        f.write("- landscape_test_2d_contour.png (2D contour plot)\n")
        f.write("- landscape_train_3d_surface.png (3D surface plot)\n")
        f.write("- landscape_train_1d_slices.png (1D slices)\n")
    
    print("Experiment log saved: loss_landscape_results/experiment_log.txt")

if __name__ == '__main__':
    main()
