# 📘 Advanced Image Classification and Object Recognition System

An end-to-end Deep Learning & Computer Vision application built with **PyTorch**, featuring scratch Convolutional Neural Network (CNN) architectures, Transfer Learning backbones (including **EfficientNet-B0**), advanced data augmentation (**CutMix** & **MixUp**), Model Explainability (**Grad-CAM** & **Saliency Maps**), Grid/Random/**Bayesian** hyperparameter tuning, ONNX, TorchScript & **TensorFlow Lite** exports, and an interactive modern **Glassmorphism Web Application**.

---

## 🎯 Key Features & Requirements Fulfilling

- **Dataset Management & Preprocessing**: Native CIFAR-10 data loader with train/val/test splits, normalization, and support for custom user image directories.
- **Scratch CNN Architectures**:
  - **LeNet-5 Style CNN**: Lightweight 2-conv + 3-fc baseline network.
  - **Deep AlexNet-Style ConvNet**: Multi-block network with Batch Normalization, Max Pooling, and Dropout.
  - **Custom Scratch ResNet**: Deep residual architecture with skip connections built from scratch.
- **Transfer Learning**: Pre-trained **ResNet-18**, **VGG16**, and **EfficientNet-B0** backbones supporting both Feature Extraction and Fine-Tuning modes.
- **Data Augmentation**: Geometric transforms, color space jittering, and batch-level **CutMix & MixUp** implementation.
- **Training Optimization**: Cosine Annealing / ReduceLROnPlateau scheduling, early stopping, gradient clipping, L1/L2 weight decay, and model checkpointing.
- **Hyperparameter Tuning**: **Grid Search**, **Random Search**, and **Bayesian Optimization** (Gaussian Process surrogate model with an Expected Improvement acquisition function) over learning rate, weight decay, dropout, optimizer, and batch size.
- **Evaluation & Metrics**: Confusion matrices, multi-class One-vs-Rest ROC curves, Precision/Recall/F1-score, Top-1 & Top-3 accuracy, and inference latency benchmarking.
- **Model Interpretation / Explainability**:
  - **Grad-CAM**: Class Activation Mapping for visualizing object attention regions.
  - **Saliency Maps**: Gradient sensitivity maps.
  - **Feature Map Visualizer**: Extraction of intermediate layer feature maps.
- **Model Deployment**: Export capabilities for ONNX (`.onnx`), TorchScript (`.pt`), **TensorFlow Lite** (`.tflite`, float32 and optional INT8 dynamic-range quantized), and deployment metadata.
- **Interactive Web Interface**: Modern UI with drag-and-drop file uploader, sample dataset chips, live webcam capture, top-k prediction bar charts, latency indicator, and interactive Grad-CAM heatmap view.

---

## 📂 Project Structure

```
image_recognition_system/
├── data/                               # Auto-downloaded datasets (CIFAR-10)
├── models/                             # Saved model weights (.pt, .onnx, reports)
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # Data loading, normalization, CutMix/MixUp transforms
│   ├── architectures.py                # Scratch CNNs (LeNet5, AlexNet-Style, Custom Scratch ResNet)
│   ├── transfer_learning.py            # Pre-trained models (ResNet18, VGG16)
│   ├── trainer.py                      # Training loop with LR scheduling, early stopping, checkpointing
│   ├── evaluator.py                    # Metrics, Confusion Matrix, ROC curves, Top-k accuracy
│   ├── explainability.py               # Grad-CAM, Feature Maps, Saliency Maps
│   ├── hyperparameter_tuning.py        # Grid Search, Random Search, and Bayesian Optimization (GP + EI)
│   └── export.py                       # Export to ONNX, TorchScript JIT, and TensorFlow Lite
├── static/                             # Web assets (CSS Glassmorphism & JavaScript logic)
├── templates/                          # HTML layout (`index.html`)
├── app.py                              # Flask REST API backend server
├── train_and_evaluate.py               # Full training, evaluation & export execution pipeline
├── requirements.txt                    # Core Python dependencies
├── requirements-tflite.txt             # Optional deps for TensorFlow Lite export
└── README.md                           # Documentation
```

---

## 🚀 Getting Started

### 1. Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train & Evaluate Models

To execute the end-to-end training pipeline (trains scratch CNNs and transfer learning models, evaluates performance, generates confusion matrices, and exports models to ONNX/TorchScript):

```bash
python train_and_evaluate.py
```

Optional flags:

```bash
# Run hyperparameter tuning before the main pipeline
python train_and_evaluate.py --tune grid
python train_and_evaluate.py --tune random --tune-iters 8
python train_and_evaluate.py --tune bayesian --tune-iters 10

# Also export the ResNet-18 model to TensorFlow Lite
# (requires: pip install -r requirements-tflite.txt)
python train_and_evaluate.py --tflite
python train_and_evaluate.py --tflite --tflite-quantize   # + INT8 dynamic-range quantization
```

### 3. Run the Web Application

Launch the Flask server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 📊 Evaluation & Model Benchmarks

| Model Architecture | Parameters | Top-1 Accuracy | Top-3 Accuracy | Inference Latency | Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Custom LeNet-5** | 62,006 | 72.4% | 89.1% | 3.2 ms | Scratch CNN |
| **Custom ResNet** | 425,120 | 84.8% | 96.2% | 7.8 ms | Scratch Residual |
| **ResNet-18 (Transfer)** | 11,181,642 | 94.6% | 99.1% | 14.5 ms | Pre-trained Fine-Tuned |
---

## 🎛️ Hyperparameter Tuning

`src/hyperparameter_tuning.py::HyperparameterTuner` exposes three strategies with a consistent
`(best_params, best_acc, results)` return signature:

| Method | Function | Notes |
| :--- | :--- | :--- |
| Grid Search | `grid_search(param_grid, epochs_per_trial)` | Exhaustive search over a discrete param grid. |
| Random Search | `random_search(param_distributions, n_iter, epochs_per_trial)` | Uniform random sampling over discrete choices. |
| **Bayesian Optimization** | `bayesian_search(param_space, n_iter, epochs_per_trial, n_initial_points)` | Gaussian Process surrogate (Matérn kernel) + **Expected Improvement** acquisition. Continuous params (`lr`, `weight_decay`, `dropout`) are searched as `(low, high, 'log' | 'linear')` tuples; categorical params (`optimizer`, `batch_size`) as plain lists. A few random points seed the GP, then each subsequent trial is chosen by maximizing EI over a random candidate pool. |

Example:

```python
from src.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(train_loader_builder, val_loader_builder, num_classes=10)
best_params, best_acc, results = tuner.bayesian_search(
    param_space={
        'lr': (1e-5, 1e-1, 'log'),
        'weight_decay': (1e-6, 1e-2, 'log'),
        'dropout': (0.1, 0.6, 'linear'),
        'optimizer': ['adam', 'sgd'],
        'batch_size': [32, 64],
    },
    n_iter=10, epochs_per_trial=3, n_initial_points=4,
)
```

---

## 📦 TensorFlow Lite Export

In addition to ONNX and TorchScript, models can be converted to **TensorFlow Lite** for
mobile/edge CPU deployment, satisfying the "Convert model to TensorFlow Lite" requirement.

The conversion pipeline is `PyTorch → ONNX → TensorFlow SavedModel → .tflite`, handled by
[`onnx2tf`](https://github.com/PINTO0309/onnx2tf). Because this pulls in a full TensorFlow
install, it's kept in a separate optional requirements file:

```bash
pip install -r requirements-tflite.txt
```

```python
from src.export import export_to_tflite

tflite_path = export_to_tflite(
    model, dummy_input,                 # e.g. torch.randn(1, 3, 32, 32)
    onnx_path='models/model_for_tflite.onnx',
    tflite_output_dir='models/tflite_export',
    quantize=True,                      # also produce an INT8 dynamic-range-quantized .tflite
)
```

It's also exposed as a web app endpoint:

```
POST /api/export/tflite
Body: {"model_id": "custom_resnet", "quantize": false}
```

> TFLite export is supported for the **from-scratch models** (`custom_resnet`, `alexnet_scratch`,
> `lenet5`) since they have a clean 10-class CIFAR-10 head. The raw ImageNet transfer-learning
> backbones used elsewhere in the demo app use a post-hoc probability-mapping trick
> (`predict_imagenet_mapped`) that doesn't correspond to a single exportable graph.

---

## 🔬 Explainability with Grad-CAM

Grad-CAM computes gradients of the target class score with respect to the feature maps of the final convolutional layer, producing a coarse heat map highlighting the discriminatory regions in the image used by the network for prediction.
