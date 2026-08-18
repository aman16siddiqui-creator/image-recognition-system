import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

class EarlyStopping:
    """
    Early stopping helper to halt training when validation metric stops improving.
    """
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class ModelTrainer:
    """
    Trainer class responsible for model training, validation, scheduling, and checkpointing.
    """
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None, clip_grad=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(self, train_loader, mixup_cutmix_fn=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if mixup_cutmix_fn is not None:
                images, targets_a, targets_b, lam = mixup_cutmix_fn(images, targets)
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            loss.backward()

            if self.clip_grad is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader, epochs=10, save_path='best_model.pt', patience=5, mixup_cutmix_fn=None):
        early_stopping = EarlyStopping(patience=patience, mode='min')
        best_val_loss = float('inf')

        print(f"Starting training on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            t_loss, t_acc = self.train_epoch(train_loader, mixup_cutmix_fn=mixup_cutmix_fn)
            v_loss, v_acc = self.evaluate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(v_loss)
                else:
                    self.scheduler.step()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['train_acc'].append(t_acc)
            self.history['val_acc'].append(v_acc)

            # Checkpoint best model
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                saved_str = "[SAVED]"
            else:
                saved_str = ""

            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f}% | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}% {saved_str}")

            if early_stopping(v_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        print(f"Training completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")
        return self.history
