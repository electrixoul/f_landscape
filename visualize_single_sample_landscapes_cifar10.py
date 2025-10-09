import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_cifar10_cnn import TinyCNN

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_data(batch_size=128):
    """Load CIFAR-10 dataset with same normalization as training"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset

def filter_normalize_direction(direction, weights):
    """Apply filter-wise normalization to direction vectors."""
    if len(weights.shape) == 4:  # Convolutional layer
        normalized = torch.zeros_like(direction)
        for i in range(weights.shape[0]):
            d_norm = torch.norm(direction[i])
            w_norm = torch.norm(weights[i])
            if d_norm > 1e-10:
                normalized[i] = (direction[i] / d_norm) * w_norm
            else:
                normalized[i] = direction[i]
        return normalized
    elif len(weights.shape) == 2:  # Fully connected layer
        normalized = torch.zeros_like(direction)
        for i in range(weights.shape[0]):
            d_norm = torch.norm(direction[i])
            w_norm = torch.norm(weights[i])
            if d_norm > 1e-10:
                normalized[i] = (direction[i] / d_norm) * w_norm
            else:
                normalized[i] = direction[i]
        return normalized
    else:
        return direction

def create_random_directions(model, ignore_bias=True):
    """Create two random direction vectors with filter-wise normalization."""
    delta = {}
    eta = {}
    
    for name, param in model.named_parameters():
        if ignore_bias and 'bias' in name:
            delta[name] = torch.zeros_like(param)
            eta[name] = torch.zeros_like(param)
        else:
            delta[name] = torch.randn_like(param)
            eta[name] = torch.randn_like(param)
            delta[name] = filter_normalize_direction(delta[name], param)
            eta[name] = filter_normalize_direction(eta[name], param)
    
    return delta, eta

def gram_schmidt(delta, eta):
    """Apply Gram-Schmidt to make directions more orthogonal."""
    inner_product = 0
    delta_norm_sq = 0
    
    for name in delta.keys():
        inner_product += torch.sum(delta[name] * eta[name]).item()
        delta_norm_sq += torch.sum(delta[name] * delta[name]).item()
    
    if delta_norm_sq > 1e-10:
        eta_ortho = {}
        for name in eta.keys():
            eta_ortho[name] = eta[name] - (inner_product / delta_norm_sq) * delta[name]
        return eta_ortho
    else:
        return eta

def filter_wise_renormalize(eta, model):
    """Re-normalize after Gram-Schmidt to restore filter-wise scaling."""
    eta_renorm = {}
    for name, param in model.named_parameters():
        if name in eta:
            eta_renorm[name] = filter_normalize_direction(eta[name], param)
    return eta_renorm

def apply_direction(model, direction, alpha):
    """Apply direction to model parameters with coefficient alpha"""
    for name, param in model.named_parameters():
        if name in direction:
            param.data.add_(direction[name], alpha=alpha)

def reset_parameters(model, original_params):
    """Reset model parameters to original values"""
    for name, param in model.named_parameters():
        param.data.copy_(original_params[name])

def precompute_logits_cnn(model, data_loader, delta, eta, device):
    """Precompute logits for CNN model."""
    model.eval()
    original_params = {name: param.data.clone() for name, param in model.named_parameters()}
    
    Z_star_list = []
    labels_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            Z_star_list.append(output.cpu())
            labels_list.append(target)
    
    Z_star = torch.cat(Z_star_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    apply_direction(model, delta, 1.0)
    Z_delta_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            Z_delta_list.append(output.cpu())
    
    Z_delta = torch.cat(Z_delta_list, dim=0)
    reset_parameters(model, original_params)
    
    apply_direction(model, eta, 1.0)
    Z_eta_list = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            Z_eta_list.append(output.cpu())
    
    Z_eta = torch.cat(Z_eta_list, dim=0)
    reset_parameters(model, original_params)
    
    Z_delta = Z_delta - Z_star
    Z_eta = Z_eta - Z_star
    
    return Z_star, Z_delta, Z_eta, labels

def sample_indices_uniform(dataset, num_samples, seed):
    """Randomly sample indices from dataset."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)
    return indices.tolist()

def compute_single_sample_loss(z_star, z_delta, z_eta, y, alpha, beta):
    """Compute loss for a single sample at given (alpha, beta)."""
    z = z_star + alpha * z_delta + beta * z_eta
    log_sum_exp = torch.logsumexp(z, dim=0)
    correct_class_logit = z[y]
    loss = log_sum_exp - correct_class_logit
    return loss.item()

def compute_single_sample_grid(z_star, z_delta, z_eta, y, alpha_vals, beta_vals):
    """Compute loss grid for a single sample using vectorized operations."""
    resolution = len(alpha_vals)
    
    A = torch.tensor(alpha_vals, dtype=torch.float32).view(resolution, 1, 1)
    B = torch.tensor(beta_vals, dtype=torch.float32).view(1, resolution, 1)
    
    z_star_bc = z_star.view(1, 1, -1)
    z_delta_bc = z_delta.view(1, 1, -1)
    z_eta_bc = z_eta.view(1, 1, -1)
    
    Z = z_star_bc + A * z_delta_bc + B * z_eta_bc
    
    log_sum_exp = torch.logsumexp(Z, dim=2)
    correct_class_logits = Z[:, :, y]
    loss_grid = log_sum_exp - correct_class_logits
    
    return loss_grid.numpy()

def plot_samples_grid(alpha_vals, beta_vals, loss_grids, metas, save_path,
                      layout=(2, 5), share_colorbar=True, color_scale='global'):
    """Plot loss landscapes for multiple samples in a grid layout."""
    rows, cols = layout
    
    if share_colorbar:
        all_values = np.concatenate([grid.flatten() for grid in loss_grids])
        if color_scale == 'global':
            vmin, vmax = all_values.min(), all_values.max()
        elif color_scale == 'quantile':
            vmin, vmax = np.percentile(all_values, [5, 95])
        else:
            vmin, vmax = all_values.min(), all_values.max()
    
    fig = plt.figure(figsize=(cols * 5, rows * 4.5))
    Alpha, Beta = np.meshgrid(alpha_vals, beta_vals)
    
    for i, (loss_grid, meta) in enumerate(zip(loss_grids, metas)):
        ax = plt.subplot(rows, cols, i + 1)
        
        if share_colorbar:
            levels = np.linspace(vmin, vmax, 30)
            contourf = ax.contourf(Alpha, Beta, loss_grid.T, levels=levels, cmap='viridis')
        else:
            contourf = ax.contourf(Alpha, Beta, loss_grid.T, levels=30, cmap='viridis')
        
        contour = ax.contour(Alpha, Beta, loss_grid.T, levels=15, colors='white',
                            alpha=0.4, linewidths=0.5)
        
        ax.plot(0, 0, marker='*', color='red', markersize=15,
                markeredgecolor='white', markeredgewidth=1)
        
        idx = meta['idx']
        y_true = meta['y_true']
        y_pred = meta['y_pred']
        prob = meta['prob']
        center_loss = meta['center_loss']
        
        title = f"Sample #{idx} | y={y_true}, ŷ={y_pred} | p={prob:.3f} | L(0,0)={center_loss:.3f}"
        ax.set_title(title, fontsize=9, fontweight='bold')
        
        ax.set_xlabel('α direction', fontsize=8)
        ax.set_ylabel('β direction', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if 'image' in meta and meta['image'] is not None:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="20%", height="20%", loc='upper right',
                             bbox_to_anchor=(0.05, 0.05, 1, 1),
                             bbox_transform=ax.transAxes, borderpad=0)
            # CIFAR-10 images are 3-channel, transpose for display
            img = np.transpose(meta['image'], (1, 2, 0))
            # Denormalize for display
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            axins.imshow(img)
            axins.axis('off')
    
    if share_colorbar:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(contourf, cax=cbar_ax)
        cbar.set_label('Loss value', fontsize=10)
    
    plt.suptitle('CIFAR-10 Single-Sample Loss Landscapes (Filter-Normalized Random Directions)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.92 if share_colorbar else 1, 0.99])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def main():
    # Configuration
    SEED = np.random.randint(0, 2**31 - 1)
    NUM_SAMPLES = 10
    SAMPLE_SET = 'train'
    RESOLUTION = 51
    ALPHA_RANGE = (-1, 1)
    BETA_RANGE = (-1, 1)
    LAYOUT = (2, 5)
    COLOR_SCALE = 'global'
    USE_GRAM_SCHMIDT = True
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("=" * 60)
    print("CIFAR-10 Single-Sample Loss Landscape Visualization")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Random seed: {SEED}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Sample set: {SAMPLE_SET}")
    print(f"Grid resolution: {RESOLUTION}x{RESOLUTION}")
    print(f"Alpha range: {ALPHA_RANGE}")
    print(f"Beta range: {BETA_RANGE}")
    print(f"Layout: {LAYOUT[0]}x{LAYOUT[1]}")
    print(f"Color scale: {COLOR_SCALE}")
    print("=" * 60)
    
    set_seed(SEED)
    os.makedirs('loss_landscape_results', exist_ok=True)
    
    print("\n1. Loading trained model...")
    model = TinyCNN().to(DEVICE)
    checkpoint = torch.load('cifar10_cnn_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Model loaded. Test accuracy: {checkpoint['test_accuracy']:.2f}%")
    
    print("\n2. Loading CIFAR-10 dataset...")
    train_loader, test_loader, train_dataset, test_dataset = load_data(batch_size=512)
    dataset = train_dataset if SAMPLE_SET == 'train' else test_dataset
    data_loader = train_loader if SAMPLE_SET == 'train' else test_loader
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Using: {SAMPLE_SET} set ({len(dataset)} samples)")
    
    print("\n3. Selecting random samples...")
    selected_indices = sample_indices_uniform(dataset, NUM_SAMPLES, SEED)
    print(f"   Selected indices: {selected_indices}")
    
    print("\n4. Creating filter-normalized random directions...")
    delta, eta = create_random_directions(model, ignore_bias=True)
    
    if USE_GRAM_SCHMIDT:
        print("   Applying Gram-Schmidt orthogonalization...")
        eta = gram_schmidt(delta, eta)
        print("   Re-normalizing eta with filter-wise normalization...")
        eta = filter_wise_renormalize(eta, model)
    
    print("   Directions created and normalized.")
    
    print(f"\n5. Precomputing logits for {SAMPLE_SET} set...")
    Z_star, Z_delta, Z_eta, labels = precompute_logits_cnn(
        model, data_loader, delta, eta, DEVICE
    )
    print(f"   Logits computed: {Z_star.shape}")
    
    print(f"\n6. Computing loss landscapes for {NUM_SAMPLES} samples...")
    alpha_vals = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], RESOLUTION)
    beta_vals = np.linspace(BETA_RANGE[0], BETA_RANGE[1], RESOLUTION)
    
    loss_grids = []
    metas = []
    
    for idx in selected_indices:
        z_star = Z_star[idx]
        z_delta = Z_delta[idx]
        z_eta = Z_eta[idx]
        y_true = labels[idx].item()
        
        loss_grid = compute_single_sample_grid(
            z_star, z_delta, z_eta, y_true, alpha_vals, beta_vals
        )
        loss_grids.append(loss_grid)
        
        center_loss = compute_single_sample_loss(z_star, z_delta, z_eta, y_true, 0, 0)
        
        logits_center = z_star.numpy()
        y_pred = logits_center.argmax()
        probs = np.exp(logits_center - np.max(logits_center))
        probs = probs / probs.sum()
        prob_true_class = probs[y_true]
        
        image, _ = dataset[idx]
        image_np = image.numpy()
        
        meta = {
            'idx': idx,
            'y_true': y_true,
            'y_pred': y_pred,
            'prob': prob_true_class,
            'center_loss': center_loss,
            'image': image_np
        }
        metas.append(meta)
        
        print(f"   Sample #{idx}: y={y_true}, ŷ={y_pred}, p={prob_true_class:.3f}, L(0,0)={center_loss:.4f}")
    
    print("\n7. Saving metadata...")
    metadata_to_save = []
    for meta in metas:
        metadata_to_save.append({
            'idx': int(meta['idx']),
            'y_true': int(meta['y_true']),
            'y_pred': int(meta['y_pred']),
            'prob': float(meta['prob']),
            'center_loss': float(meta['center_loss'])
        })
    
    with open('loss_landscape_results/selected_samples_cifar10.json', 'w') as f:
        json.dump({
            'seed': SEED,
            'sample_set': SAMPLE_SET,
            'num_samples': NUM_SAMPLES,
            'indices': selected_indices,
            'samples': metadata_to_save
        }, f, indent=2)
    print("   Saved: loss_landscape_results/selected_samples_cifar10.json")
    
    print("\n8. Creating visualization...")
    plot_samples_grid(
        alpha_vals, beta_vals, loss_grids, metas,
        'loss_landscape_results/samples_landscape_grid_cifar10.png',
        layout=LAYOUT,
        share_colorbar=True,
        color_scale=COLOR_SCALE
    )
    
    plot_samples_grid(
        alpha_vals, beta_vals, loss_grids, metas,
        'loss_landscape_results/samples_landscape_grid_cifar10.pdf',
        layout=LAYOUT,
        share_colorbar=True,
        color_scale=COLOR_SCALE
    )
    
    print("\n9. Saving individual sample data...")
    for i, (idx, loss_grid) in enumerate(zip(selected_indices, loss_grids)):
        np.savez(f'loss_landscape_results/sample_{idx}_loss_grid_cifar10.npz',
                 alpha=alpha_vals,
                 beta=beta_vals,
                 loss=loss_grid,
                 idx=idx,
                 y_true=metas[i]['y_true'],
                 center_loss=metas[i]['center_loss'])
    print(f"   Saved {NUM_SAMPLES} individual sample grids")
    
    print("\n" + "=" * 60)
    print("Single-sample loss landscape analysis completed!")
    print("Results saved in: loss_landscape_results/")
    print("=" * 60)
    
    with open('loss_landscape_results/single_sample_experiment_log_cifar10.txt', 'w') as f:
        f.write("CIFAR-10 Single-Sample Loss Landscape Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Random seed: {SEED}\n")
        f.write(f"Number of samples: {NUM_SAMPLES}\n")
        f.write(f"Sample set: {SAMPLE_SET}\n")
        f.write(f"Grid resolution: {RESOLUTION}x{RESOLUTION}\n")
        f.write(f"Alpha range: {ALPHA_RANGE}\n")
        f.write(f"Beta range: {BETA_RANGE}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Normalization method: Filter-wise\n")
        f.write(f"Bias handling: Ignored (set to zero)\n")
        f.write(f"Gram-Schmidt: {USE_GRAM_SCHMIDT}\n")
        f.write(f"Color scale: {COLOR_SCALE}\n")
        f.write(f"Layout: {LAYOUT[0]}x{LAYOUT[1]}\n\n")
        f.write(f"Model test accuracy: {checkpoint['test_accuracy']:.2f}%\n\n")
        f.write("Selected samples:\n")
        for meta in metadata_to_save:
            f.write(f"  Sample #{meta['idx']}: y={meta['y_true']}, ŷ={meta['y_pred']}, "
                   f"p={meta['prob']:.3f}, L(0,0)={meta['center_loss']:.4f}\n")
        f.write("\nFiles generated:\n")
        f.write("- selected_samples_cifar10.json (sample metadata)\n")
        f.write("- samples_landscape_grid_cifar10.png (tiled visualization)\n")
        f.write("- samples_landscape_grid_cifar10.pdf (tiled visualization, PDF)\n")
        f.write(f"- sample_<idx>_loss_grid_cifar10.npz (individual grids, {NUM_SAMPLES} files)\n")
    
    print("Experiment log saved: loss_landscape_results/single_sample_experiment_log_cifar10.txt")

if __name__ == '__main__':
    main()
