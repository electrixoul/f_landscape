import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define the minimal linear classifier network
class LinearMNIST(nn.Module):
    def __init__(self):
        super(LinearMNIST, self).__init__()
        # First layer: input size = 28*28 = 784, output size = 10
        self.fc = nn.Linear(784, 10)
    
    def forward(self, x):
        # Flatten the input image
        x = x.view(-1, 784)
        # Apply linear transformation
        x = self.fc(x)
        return x

def train():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.01
    epochs = 10
    
    # Set device - use MPS for M-series Mac, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    # Create data directory
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = LinearMNIST().to(device)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/(batch_idx+1):.4f}, "
                      f"Accuracy: {100*correct/total:.2f}%")
        
        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Summary - Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_acc:.2f}%")
        
        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        print("-" * 60)
    
    # Save the model
    model_path = 'mnist_linear_model.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': test_acc,
    }, model_path)
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    train()
