import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Evaluation tool for classification accuracy, confusion matrix, ROC curves, and latency benchmarking.
    """
    def __init__(self, model, class_names, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names

    def get_predictions(self, data_loader):
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        return all_probs, all_preds, all_targets

    def compute_topk_accuracy(self, all_probs, all_targets, k=3):
        topk_preds = np.argsort(all_probs, axis=1)[:, -k:]
        correct = np.any(topk_preds == all_targets[:, None], axis=1)
        topk_acc = 100.0 * np.mean(correct)
        top1_acc = 100.0 * np.mean(np.argmax(all_probs, axis=1) == all_targets)
        return top1_acc, topk_acc

    def compute_metrics(self, data_loader, top_k=3):
        probs, preds, targets = self.get_predictions(data_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        top1_acc, topk_acc = self.compute_topk_accuracy(probs, targets, k=top_k)

        report_dict = classification_report(targets, preds, target_names=self.class_names, output_dict=True)

        return {
            'top1_accuracy': top1_acc,
            'topk_accuracy': topk_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report_dict,
            'confusion_matrix': confusion_matrix(targets, preds).tolist(),
            'probs': probs,
            'preds': preds,
            'targets': targets
        }

    def benchmark_latency(self, sample_tensor, num_runs=100):
        """
        Measures mean inference time per image in milliseconds.
        """
        sample_tensor = sample_tensor.to(self.device)
        if sample_tensor.dim() == 3:
            sample_tensor = sample_tensor.unsqueeze(0)

        # Warmup
        for _ in range(10):
            _ = self.model(sample_tensor)

        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(sample_tensor)
        end = time.time()

        avg_latency_ms = ((end - start) / num_runs) * 1000.0
        return avg_latency_ms

    def plot_confusion_matrix(self, cm, save_path=None):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        return fig

    def plot_roc_curves(self, probs, targets, save_path=None):
        num_classes = len(self.class_names)
        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(num_classes):
            binary_target = (targets == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_target, probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multi-Class One-vs-Rest ROC Curves')
        ax.legend(loc='lower right', fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        return fig
