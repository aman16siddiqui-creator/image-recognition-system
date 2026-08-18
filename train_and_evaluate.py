import sys
import os
import time
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

from src.dataset import load_cifar10_data, MixUpCutMixBatchTransform, CIFAR10_CLASSES
from src.architectures import LeNet5, DeepAlexNetStyleCNN, CustomScratchResNet
from src.transfer_learning import get_pretrained_model, print_parameter_summary
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.explainability import GradCAM, apply_heatmap_overlay, extract_feature_maps
from src.export import export_to_onnx, export_to_torchscript, export_summary

def main():
    print("=" * 70)
    print("[START] Project 7: Image Classification & Object Recognition Pipeline")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on Device: {device}")

    # 1. Dataset Loading
    data_dir = './data'
    batch_size = 64
    epochs = 3  # Fast demonstration training run (can be adjusted)
    
    print("\n[Phase 1] Loading CIFAR-10 Dataset & Augmentation Pipeline...")
    train_loader, val_loader, test_loader, classes = load_cifar10_data(data_dir=data_dir, batch_size=batch_size)
    print(f"Loaded {len(train_loader.dataset)} training, {len(val_loader.dataset)} validation, {len(test_loader.dataset)} test samples.")
    print(f"Classes ({len(classes)}): {classes}")

    mixup_cutmix_fn = MixUpCutMixBatchTransform(mixup_alpha=1.0, cutmix_alpha=1.0, prob=0.3)

    models_dict = {}
    benchmark_results = {}

    # 2. Scratch CNN 1: LeNet-5
    print("\n[Phase 2] Training Scratch CNN 1: LeNet-5...")
    lenet = LeNet5(num_classes=10).to(device)
    opt_lenet = torch.optim.Adam(lenet.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer_lenet = ModelTrainer(lenet, opt_lenet, nn.CrossEntropyLoss(), device=device)
    trainer_lenet.fit(train_loader, val_loader, epochs=epochs, save_path='models/custom_lenet.pt')
    models_dict['Custom LeNet-5'] = lenet

    # 3. Scratch CNN 2: Custom Scratch ResNet
    print("\n[Phase 3] Training Scratch CNN 2: Custom Scratch ResNet...")
    scratch_resnet = CustomScratchResNet(num_classes=10).to(device)
    opt_sresnet = torch.optim.Adam(scratch_resnet.parameters(), lr=1e-3, weight_decay=1e-4)
    trainer_sresnet = ModelTrainer(scratch_resnet, opt_sresnet, nn.CrossEntropyLoss(), device=device)
    trainer_sresnet.fit(train_loader, val_loader, epochs=epochs, save_path='models/custom_resnet.pt', mixup_cutmix_fn=mixup_cutmix_fn)
    models_dict['Custom ResNet'] = scratch_resnet

    # 4. Transfer Learning Model: ResNet-18
    print("\n[Phase 4] Fine-Tuning Transfer Learning Model: ResNet-18...")
    resnet18 = get_pretrained_model(model_name='resnet18', num_classes=10, feature_extract=False, pretrained=True).to(device)
    print_parameter_summary(resnet18)
    opt_r18 = torch.optim.Adam(resnet18.parameters(), lr=3e-4, weight_decay=1e-4)
    trainer_r18 = ModelTrainer(resnet18, opt_r18, nn.CrossEntropyLoss(), device=device)
    trainer_r18.fit(train_loader, val_loader, epochs=epochs, save_path='models/resnet18_transfer.pt')
    models_dict['ResNet-18 (Transfer)'] = resnet18

    # 5. Evaluation & Benchmarking
    print("\n[Phase 5] Evaluating Models & Latency Benchmarking...")
    os.makedirs('models/reports', exist_ok=True)
    sample_tensor = torch.randn(1, 3, 32, 32)

    for name, model in models_dict.items():
        evaluator = ModelEvaluator(model, classes, device=device)
        metrics = evaluator.compute_metrics(test_loader, top_k=3)
        latency_ms = evaluator.benchmark_latency(sample_tensor)

        benchmark_results[name] = {
            'top1_acc': metrics['top1_accuracy'],
            'top3_acc': metrics['topk_accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'latency_ms': latency_ms
        }

        print(f"📊 {name} -> Top-1 Acc: {metrics['top1_accuracy']:.2f}% | Top-3 Acc: {metrics['topk_accuracy']:.2f}% | Latency: {latency_ms:.2f} ms/img")

        # Save confusion matrix and ROC curves
        evaluator.plot_confusion_matrix(metrics['confusion_matrix'], save_path=f"models/reports/{name.replace(' ', '_').lower()}_cm.png")

    # 6. Model Interpretation (Grad-CAM test)
    print("\n[Phase 6] Testing Grad-CAM Explainability...")
    sample_img, sample_lbl = test_loader.dataset[0]
    sample_tensor = sample_img.unsqueeze(0).to(device)

    # Use ResNet18 layer4 for Grad-CAM
    gradcam = GradCAM(resnet18, resnet18.layer4)
    heatmap, pred_cls = gradcam.generate(sample_tensor)
    print(f"Grad-CAM generated for target class: {classes[pred_cls]} (True class: {classes[sample_lbl]})")
    gradcam.remove_hooks()

    # 7. Model Exporting
    print("\n[Phase 7] Exporting Models for Deployment...")
    export_to_onnx(resnet18, sample_tensor.cpu(), output_path='models/resnet18.onnx')
    export_to_torchscript(resnet18, sample_tensor.cpu(), output_path='models/resnet18_jit.pt')
    export_summary(resnet18, save_path='models/export_summary.txt')

    print("\n✅ Training & Evaluation Pipeline Completed Successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
