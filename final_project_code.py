import torch  
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from random import random
from PIL import Image
import pandas as pd
import os

transform = Compose([
    Resize((224, 224)),
    ToTensor()
])


# TODO: change this to a .env file for easier use

root ='/Users/adelinetan/Documents/ECE176/ECE176_final_project/bcn20000'

dataset = ImageFolder(root='BCN_20k_test\BCN_20k_train', transform=transform)

print(f"Classes found: {dataset.classes}")   # prints your disease names
print(f"Total images:  {len(dataset)}")

# Define only the valid disease classes to use
CLASSES = [
    'melanoma',
    'nevus',
    'basal_cell_carcinoma',
    'actinic_keratosis',
    'benign_keratosis',
    'dermatofibroma',
    'vascular_lesion',
    'squamous_cell_carcinoma'
]

dataset = ImageFolder(
    root='../bcn20000',
    transform=transform,
    is_valid_file=lambda path: (
        path.lower().endswith(('.jpg', '.jpeg', '.png')) and
        any(cls in path for cls in CLASSES)  # skip 'licenses' folder
    )
)

print(f"Classes found: {dataset.classes}")
print(f"Total images: {len(dataset)}")


total  = len(dataset)
n_train = int(0.7 * total)
n_val   = int(0.2 * total)
n_test  = total - n_train - n_val   

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=2)

images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}")        # [32, 3, 224, 224]
print(f"Labels:      {labels}")              # tensor of class indices

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(to_pil_image(images[i]))
    ax.set_title(dataset.classes[labels[i]])
    ax.axis('off')
plt.tight_layout()
plt.show()