import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
import os
from train_cifar10_cnn import TinyCNN

def load_failed_samples():
    """Load indices of misclassified samples from the visualization run"""
    with open('loss_landscape_results/selected_samples_cifar100.json', 'r') as f:
        data = json.load(f)
    
    # Extract indices where y_true != y_pred
    failed_indices = []
    for sample in data['samples']:
        if sample['y_true'] != sample['y_pred']:
            failed_indices.append(sample['idx'])
    
    return failed_indices, data['sample_set']

def create_fail_dataset(failed_indices, sample_set='test'):
    """Create a small dataset containing only the failed samples"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    if sample_set == 'test':
        full_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=False,
            transform=transform
        )
    else:
        full_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=False,
            transform=transform
        )
    
    fail_dataset = Subset(full_dataset, failed_indices)
    return fail_dataset, full_dataset

def train_overfit():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
    print("=" * 60)
    print("Training to Overfit on Failed Samples")
    print("=" * 60)
    
    # Load failed sample indices
    print("\n1. Loading failed sample indices...")
    failed_indices, sample_set = load_failed_samples()
    print(f"   Found {len(failed_indices)} misclassified samples from {sample_set} set")
    print(f"   Failed sample indices: {failed_indices}")
    
    if len(failed_indices) == 0:
        print("   No failed samples found! All samples were correctly classified.")
        return
    
    # Create fail dataset
    print("\n2. Creating fail dataset...")
    fail_dataset, full_dataset = create_fail_dataset(failed_indices, sample_set)
    fail_loader = DataLoader(fail_dataset, batch_size=min(len(fail_dataset), 32), shuffle=True)
    
    # Also create a loader for evaluation on full test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"   Fail dataset size: {len(fail_dataset)} samples")
    
    # Load pre-trained model
    print("\n3. Loading pre-trained model...")
    model = TinyCNN().to(device)
    checkpoint = torch.load('cifar10_cnn_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    initial_test_acc = checkpoint['test_accuracy']
    print(f"   Model loaded. Initial test accuracy: {initial_test_acc:.2f}%")
    
    # Evaluate initial accuracy on fail dataset
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in fail_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        initial_fail_acc = 100 * correct / total
    print(f"   Initial accuracy on fail dataset: {initial_fail_acc:.2f}%")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Lower LR for fine-tuning
    
    print("\n4. Starting overfitting training...")
    print("   Training until 100% accuracy on fail dataset...")
    print("-" * 60)
    
    epoch = 0
    max_epochs = 1000
    fail_acc = initial_fail_acc
    
    while fail_acc < 100.0 and epoch < max_epochs:
        epoch += 1
        model.train()
        running_loss = 0.0
        
        for data, target in fail_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Evaluate on fail dataset
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in fail_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            fail_acc = 100 * correct / total
        
        avg_loss = running_loss / len(fail_loader)
        
        if epoch % 10 == 0 or fail_acc == 100.0:
            # Also evaluate on full test set
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    test_total += target.size(0)
                    test_correct += (predicted == target).sum().item()
                test_acc = 100 * test_correct / test_total
            
            print(f"Epoch [{epoch:4d}] Loss: {avg_loss:.4f}, "
                  f"Fail Acc: {fail_acc:.2f}%, Test Acc: {test_acc:.2f}%")
    
    print("-" * 60)
    print(f"\n5. Training completed after {epoch} epochs")
    print(f"   Final accuracy on fail dataset: {fail_acc:.2f}%")
    
    # Evaluate final model on full test set
    model.eval()
    with torch.no_grad():
        test_correct = 0
        test_total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
        final_test_acc = 100 * test_correct / test_total
    
    print(f"   Final test accuracy on full dataset: {final_test_acc:.2f}%")
    
    # Save overfitted model
    print("\n6. Saving overfitted model...")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_accuracy': final_test_acc,
        'fail_dataset_accuracy': fail_acc,
        'fail_indices': failed_indices,
        'initial_test_accuracy': initial_test_acc,
    }, 'cifar100_cnn_overfitted.pth')
    print("   Model saved to: cifar100_cnn_overfitted.pth")
    
    # Save training log
    with open('loss_landscape_results/overfitting_training_log.txt', 'w') as f:
        f.write("CIFAR-100 Overfitting Training on Failed Samples\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Failed samples: {failed_indices}\n")
        f.write(f"Number of failed samples: {len(failed_indices)}\n")
        f.write(f"Sample set: {sample_set}\n\n")
        f.write(f"Training epochs: {epoch}\n")
        f.write(f"Initial test accuracy: {initial_test_acc:.2f}%\n")
        f.write(f"Final test accuracy: {final_test_acc:.2f}%\n")
        f.write(f"Accuracy change: {final_test_acc - initial_test_acc:+.2f}%\n\n")
        f.write(f"Initial accuracy on fail dataset: {initial_fail_acc:.2f}%\n")
        f.write(f"Final accuracy on fail dataset: {fail_acc:.2f}%\n\n")
        f.write("Model saved to: cifar100_cnn_overfitted.pth\n")
    
    print("   Training log saved: loss_landscape_results/overfitting_training_log.txt")
    print("\n" + "=" * 60)
    print("Overfitting training completed!")
    print("=" * 60)

if __name__ == '__main__':
    train_overfit()
