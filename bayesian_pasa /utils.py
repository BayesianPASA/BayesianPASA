import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import numpy as np
import random
import os
from PIL import Image, ImageFilter
from tqdm import tqdm
from copy import deepcopy


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar100_loaders(batch_size=128):
    """Return train and test loaders for clean CIFAR-100"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return trainloader, testloader


class CachedNoisyCIFAR10(Dataset):
    """
    Dataset with in-memory caching for corrupted CIFAR-10 images.
    """
    def __init__(self, noise_type='gaussian', severity=3, num_samples=None, train=True):
        self.noise_type = noise_type
        self.severity = severity
        self.train = train
        self.clean_dataset = CIFAR10(root='./data', train=train, download=True, transform=None)
        
        if num_samples is None:
            self.num_samples = len(self.clean_dataset)
        else:
            self.num_samples = min(num_samples, len(self.clean_dataset))

        self.cached_samples = []
        self.cached_labels = []

        indices = list(range(self.num_samples))
        print(f"Caching {noise_type} samples ({self.num_samples} images)...")
        for idx in tqdm(indices, leave=False):
            img, label = self.clean_dataset[idx]
            img_tensor = self._apply_noise_and_transform(img)
            self.cached_samples.append(img_tensor)
            self.cached_labels.append(label)

    def _apply_noise_and_transform(self, img_pil):
        img_np = np.array(img_pil).astype(np.float32) / 255.0

        if self.noise_type == 'gaussian':
            noise = np.random.randn(*img_np.shape) * 0.1 * self.severity
            img_np = img_np + noise
        elif self.noise_type == 'shot_noise':
            mask = np.random.random(img_np.shape) < 0.05 * self.severity
            salt = np.random.random(mask.sum()) > 0.5
            img_np_flat = img_np.reshape(-1)
            mask_flat = mask.reshape(-1)
            img_np_flat[mask_flat] = salt.astype(np.float32)
            img_np = img_np_flat.reshape(img_np.shape)
        elif self.noise_type == 'blur':
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=self.severity*0.5))
            img_np = np.array(img_pil).astype(np.float32) / 255.0
        elif self.noise_type == 'contrast':
            mean = img_np.mean()
            contrast_factor = max(0.5, 1.0 - 0.2 * self.severity)
            img_np = contrast_factor * (img_np - mean) + mean

        img_np = np.clip(img_np, 0, 1)
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float()

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        img_tensor = normalize(img_tensor)
        return img_tensor

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        return self.cached_samples[idx], self.cached_labels[idx]


def get_cifar10c_loaders(noise_type='gaussian', batch_size=64, 
                         train_samples=12500, test_samples=2500):
    """Return train and test loaders for corrupted CIFAR-10"""
    trainset = CachedNoisyCIFAR10(noise_type=noise_type, severity=3,
                                   num_samples=train_samples, train=True)
    testset = CachedNoisyCIFAR10(noise_type=noise_type, severity=3,
                                  num_samples=test_samples, train=False)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    return trainloader, testloader


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_and_evaluate(model, trainloader, testloader, epochs=50, lr=0.1, device='cuda'):
    """Complete training and evaluation loop"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()
        
        test_accs.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    smoothed_acc = sum(test_accs[-5:]) / 5
    return best_acc, smoothed_acc
