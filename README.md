# f_landscape

A minimal MNIST linear classification network implementation using PyTorch with loss landscape visualization.

## Project Description

This project implements the simplest possible neural network for MNIST digit classification:
- A single linear layer (784 → 10) with no hidden layers
- Cross-entropy loss function
- Standard backpropagation training
- Optimized for Apple Silicon (M4 Pro) with MPS support
- Loss landscape visualization using filter-wise normalization (following Li et al., NeurIPS 2018)

## Network Architecture

```
Input: 28x28 MNIST image (flattened to 784 dimensions)
  ↓
Linear Layer (784 → 10)
  ↓
Output: 10 class scores (argmax for prediction)
```

Total parameters: 7,850 (784 × 10 weights + 10 biases)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- NumPy 1.20+
- Matplotlib 3.5+

## Setup

1. Activate the conda AI environment:
```bash
source ~/.bash_profile
conda activate ai_env
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script:
```bash
python train_mnist.py
```

The script will:
- Automatically download the MNIST dataset to `./data/` directory
- Train the model for 10 epochs with batch size of 64
- Use SGD optimizer with learning rate 0.01
- Display training progress and accuracy every 100 batches
- Evaluate on test set after each epoch
- Save the trained model to `mnist_linear_model.pth`

### Loss Landscape Visualization

#### 1. Overall Loss Landscape

Visualize the overall loss landscape (averaged over entire dataset):
```bash
python visualize_loss_landscape.py
```

This script implements the visualization method from "Visualizing the Loss Landscape of Neural Nets" (Li et al., NeurIPS 2018):
- Creates two random direction vectors with **filter-wise normalization**
- Ignores bias parameters (sets perturbation to zero)
- Computes loss over a 51×51 grid in the 2D plane defined by these directions
- Generates multiple visualization outputs

The script will generate:
- `landscape_train_2d_contour.png` - 2D contour plot of training loss
- `landscape_test_2d_contour.png` - 2D contour plot of test loss
- `landscape_train_3d_surface.png` - 3D surface plot
- `landscape_train_1d_slices.png` - 1D slices along α and β axes
- `loss_grid_train.npz` - Numerical data for training set
- `loss_grid_test.npz` - Numerical data for test set
- `experiment_log.txt` - Experimental parameters and results

#### 2. Single-Sample Loss Landscapes

Visualize loss landscapes for individual samples:
```bash
python visualize_single_sample_landscapes.py
```

This script extends the methodology to analyze individual samples:
- Randomly selects 10 samples from the dataset
- Computes the loss landscape for each sample independently
- Uses the same filter-wise normalized directions for all samples
- Creates a tiled visualization for easy comparison

The script will generate:
- `samples_landscape_grid.png` - Tiled 2D contour plots (2×5 layout)
- `samples_landscape_grid.pdf` - PDF version of tiled plot
- `selected_samples.json` - Metadata for selected samples
- `sample_<idx>_loss_grid.npz` - Individual sample loss grids (10 files)
- `single_sample_experiment_log.txt` - Experiment log

#### Key Features of the Visualization:
- **Filter-wise normalization**: Each "filter" (output neuron weight vector) in the random directions is normalized to have the same norm as the corresponding filter in the trained model
- **Bias ignored**: Following the paper's recommendation, bias parameters are not perturbed
- **Efficient computation**: Logits are precomputed once, then linearly combined for each grid point
- **Reproducible**: Fixed random seed (42) for consistent results
- **Sample-level analysis**: Reveals how individual samples contribute to the overall loss landscape
- **Shared colorbar**: Enables direct comparison across samples

### Loading the Trained Model

```python
import torch
from train_mnist import LinearMNIST

# Load the model
model = LinearMNIST()
checkpoint = torch.load('mnist_linear_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use the model for inference
# ...
```

## Hyperparameters

### Training
- Batch size: 64
- Learning rate: 0.01
- Epochs: 10
- Optimizer: SGD (vanilla backpropagation)

### Loss Landscape Visualization
- Grid resolution: 51×51 (default from paper)
- Alpha range: [-1, 1]
- Beta range: [-1, 1]
- Random seed: 42
- Normalization: Filter-wise
- Bias handling: Ignored (set to zero)

## Expected Performance

### MNIST Linear Classifier
This minimal linear classifier typically achieves around 92-93% accuracy on the MNIST test set, demonstrating that even without hidden layers or non-linearities, a simple linear model can perform reasonably well on this dataset.

The loss landscape visualization reveals:
- A relatively smooth and convex loss surface
- The trained model (θ*) sits at the center (0, 0) in a local minimum
- Loss increases gradually in all directions from the minimum

### CIFAR-10 TinyCNN
The minimal CNN achieves around 73-74% test accuracy on CIFAR-10 with only 19,466 parameters (2.5x the MNIST linear model).

The loss landscape visualization reveals:
- More complex non-convex structure compared to MNIST linear classifier
- Presence of local minima and saddle points
- Steeper loss increase in certain directions
- Filter-wise normalization enables meaningful comparison across different architectures

## CIFAR-10 CNN Experiments

### Training CIFAR-10 Model

```bash
python train_cifar10_cnn.py
```

Trains a minimal CNN (19,466 parameters) on CIFAR-10:
- TinyCNN architecture: 3 Conv layers + 1 FC layer
- 50 epochs with SGD (momentum 0.9)
- Learning rate: 0.01 with MultiStepLR scheduler
- Saves best model to `cifar10_cnn_model.pth`

### CIFAR-10 Loss Landscape Visualization

**Overall landscape:**
```bash
python visualize_loss_landscape_cifar10.py
```

**Single-sample landscapes:**
```bash
python visualize_single_sample_landscapes_cifar10.py
```

These scripts work identically to the MNIST versions but are adapted for:
- CNN architecture with convolutional layers
- CIFAR-10 dataset (32×32 RGB images)
- Filter-wise normalization for 4D convolutional filters

## File Structure

```
f_landscape/
├── train_mnist.py                          # MNIST training script
├── train_cifar10_cnn.py                    # CIFAR-10 CNN training script
├── visualize_loss_landscape.py             # MNIST loss landscape
├── visualize_loss_landscape_cifar10.py     # CIFAR-10 loss landscape
├── visualize_single_sample_landscapes.py   # MNIST single-sample landscapes
├── visualize_single_sample_landscapes_cifar10.py  # CIFAR-10 single-sample
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
├── .gitignore                             # Git ignore rules
├── mnist_linear_loss_landscape_plan.md    # MNIST visualization plan
├── plan_single_sample_loss_landscapes.md  # Single-sample plan
├── Visualizing the Loss Landscape of Neural Nets.pdf  # Reference paper
├── data/                                  # Datasets (gitignored)
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── mnist_linear_model.pth                 # MNIST trained model (gitignored)
├── cifar10_cnn_model.pth                  # CIFAR-10 trained model (gitignored)
└── loss_landscape_results/                # Visualization outputs (gitignored)
    ├── MNIST results (landscape_*.png, loss_grid_*.npz, etc.)
    └── CIFAR-10 results (*_cifar10.png, *_cifar10.npz, etc.)
```

## References

The loss landscape visualization methodology is based on:

**Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).** *Visualizing the Loss Landscape of Neural Nets.* In Advances in Neural Information Processing Systems (NeurIPS 2018).

Key concepts implemented:
- Filter-wise normalization for meaningful scale-invariant comparisons
- 2D loss surface slicing using random directions
- Efficient computation via logit precomputation for linear models
