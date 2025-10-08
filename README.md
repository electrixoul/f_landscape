# f_landscape

A minimal MNIST linear classification network implementation using PyTorch.

## Project Description

This project implements the simplest possible neural network for MNIST digit classification:
- A single linear layer (784 → 10) with no hidden layers
- Cross-entropy loss function
- Standard backpropagation training
- Optimized for Apple Silicon (M4 Pro) with MPS support

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

- Batch size: 64
- Learning rate: 0.01
- Epochs: 10
- Optimizer: SGD (vanilla backpropagation)

## Expected Performance

This minimal linear classifier typically achieves around 92-93% accuracy on the MNIST test set, demonstrating that even without hidden layers or non-linearities, a simple linear model can perform reasonably well on this dataset.

## File Structure

```
f_landscape/
├── train_mnist.py          # Main training script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore            # Git ignore rules
├── data/                 # MNIST dataset (auto-downloaded, gitignored)
└── mnist_linear_model.pth # Trained model (gitignored)
