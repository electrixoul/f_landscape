import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Define a CNN classifier for CIFAR-100 (5x larger than original TinyCNN)
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Convolutional layers (5x larger)
        self.conv1 = nn.Conv2d(3, 22, kernel_size=3, padding=1)  # 3->22: 616 params
        self.conv2 = nn.Conv2d(22, 44, kernel_size=3, padding=1)  # 22->44: 8,756 params
        self.conv3 = nn.Conv2d(44, 44, kernel_size=3, padding=1)  # 44->44: 17,468 params
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        # Output: 44 channels x 4x4 = 704
        self.fc = nn.Linear(44 * 4 * 4, 100)  # 704->100: 70,500 params
        
        # Total parameters: 616 + 8,756 + 17,468 + 70,500 = 97,340 (~5x)
    
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)  # Flatten to (batch, 512)
        x = self.fc(x)
        
        return x

def train():
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.01
    epochs = 50
    
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
    
    # Data transforms for CIFAR-100
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # Load CIFAR-100 dataset
    print("Loading CIFAR-100 dataset...")
    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
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
    model = TinyCNN().to(device)
    print(f"\nModel architecture:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    best_test_acc = 0.0
    
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
        
        # Update learning rate
        scheduler.step()
        
        # Epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Summary - Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}")
        
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
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
                'best_test_accuracy': best_test_acc,
            }, 'cifar10_cnn_model.pth')
            print(f"*** New best model saved with test accuracy: {test_acc:.2f}% ***")
    
    print(f"\nTraining completed!")
    print(f"Model saved to: cifar10_cnn_model.pth")
    print(f"Best test accuracy: {best_test_acc:.2f}%")

if __name__ == '__main__':
    train()
