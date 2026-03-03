import torch
import torchvision
import torchvision.transforms as transforms

from models.mlp import MLP
from train.backprop_train import train_backprop
from ga.genetic import train_genetic
from evaluate import evaluate

# --- 1. Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# --- 2. Backpropagation ---
model_bp = MLP()
train_backprop(model_bp, train_loader, epochs=5)
acc_bp = evaluate(model_bp, test_loader)

# --- 3. Algorithme Génétique ---
model_ga = MLP()
train_genetic(model_ga, train_loader, generations=20, pop_size=30, select_k=10)
acc_ga = evaluate(model_ga, test_loader)

print(f"Backprop accuracy: {acc_bp:.2f}%")
print(f"GA accuracy: {acc_ga:.2f}%")