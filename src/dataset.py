import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from PIL import Image

# CIFAR-10 Default Class Names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Normalization constants (CIFAR-10 standard)
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2470, 0.2435, 0.2616]


def get_transforms(img_size=(32, 32), augment=True):
    """
    Constructs train and test transformation pipelines.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, test_transform


class MixUpCutMixBatchTransform:
    """
    Batch-level implementation of MixUp and CutMix data augmentation.
    """
    def __init__(self, mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x, y):
        if np.random.rand() > self.prob:
            return x, y, y, 1.0

        use_cutmix = np.random.rand() > 0.5
        batch_size = x.size(0)
        index = torch.randperm(batch_size)

        if use_cutmix and self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            return x, y, y[index], lam
        elif self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            mixed_x = lam * x + (1 - lam) * x[index]
            return mixed_x, y, y[index], lam
        return x, y, y, 1.0


def load_cifar10_data(data_dir='./data', batch_size=64, img_size=(32, 32), num_workers=0, val_split=0.1):
    """
    Downloads and prepares CIFAR-10 data loaders (Train, Validation, Test).
    """
    train_tf, test_tf = get_transforms(img_size=img_size, augment=True)

    full_train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    # Train / Validation Split
    val_size = int(len(full_train_ds) * val_split)
    train_size = len(full_train_ds) - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size], generator=generator)

    # Set evaluation transform on validation split
    val_ds.dataset.transform = test_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, CIFAR10_CLASSES


class CustomImageDirectoryDataset(Dataset):
    """
    Custom Dataset loader for user-provided image directories structured as:
    root_dir/
      class_a/
        img1.jpg
      class_b/
        img2.png
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
