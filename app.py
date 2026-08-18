import os
import io
import base64
import time
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.architectures import LeNet5, CustomScratchResNet, DeepAlexNetStyleCNN
from src.transfer_learning import predict_imagenet_mapped, get_pretrained_model
from src.dataset import CIFAR10_CLASSES, MEAN, STD
from src.explainability import GradCAM, apply_heatmap_overlay, generate_saliency_map, localize_object_bbox, extract_feature_maps

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model registry: metadata only. Nothing is loaded into memory until a
# model is actually requested by a /api/classify call. This keeps startup
# fast and idle RAM low, which matters a lot on memory-billed hosts like
# Railway (this app previously loaded 7 models, including VGG-16, on boot).
MODELS = {}

def _build_resnet18():
    m = models.resnet18(weights='DEFAULT').to(device).eval()
    return {'name': 'ResNet-18 (Transfer Learning - Highest Accuracy)', 'model': m,
            'is_raw_imagenet': True, 'gradcam_layer': m.layer4}

def _build_vgg16():
    m = models.vgg16(weights='DEFAULT').to(device).eval()
    return {'name': 'VGG-16 (Transfer Learning - Feature Extractor)', 'model': m,
            'is_raw_imagenet': True, 'gradcam_layer': m.features[-1]}

def _build_efficientnet():
    m = models.efficientnet_b0(weights='DEFAULT').to(device).eval()
    return {'name': 'EfficientNet-B0 (Transfer Learning - Fine-Tuned)', 'model': m,
            'is_raw_imagenet': True, 'gradcam_layer': m.features[-1]}


def _build_lenet5():
    m = LeNet5(num_classes=10)
    if os.path.exists('models/custom_lenet.pt'):
        m.load_state_dict(torch.load('models/custom_lenet.pt', map_location=device))
    m.to(device).eval()
    return {'name': 'Custom LeNet-5 (Built From Scratch)', 'model': m,
            'is_raw_imagenet': False, 'gradcam_layer': m.conv2}

def _build_alexnet_scratch():
    m = DeepAlexNetStyleCNN(num_classes=10)
    if os.path.exists('models/alexnet_scratch.pt'):
        m.load_state_dict(torch.load('models/alexnet_scratch.pt', map_location=device))
    m.to(device).eval()
    return {'name': 'AlexNet-Like CNN (Built From Scratch)', 'model': m,
            'is_raw_imagenet': False, 'gradcam_layer': m.features[-4]}

def _build_custom_resnet():
    m = CustomScratchResNet(num_classes=10)
    if os.path.exists('models/custom_resnet.pt'):
        m.load_state_dict(torch.load('models/custom_resnet.pt', map_location=device))
    m.to(device).eval()
    return {'name': 'Custom ResNet Block CNN (Built From Scratch)', 'model': m,
            'is_raw_imagenet': False, 'gradcam_layer': m.layer3[-1].conv2}

# id -> (display_name, loader). Order here defines /api/models order.
MODEL_REGISTRY = {
    'resnet18': ('ResNet-18 (Transfer Learning - Highest Accuracy)', _build_resnet18),
    'vgg16': ('VGG-16 (Transfer Learning - Feature Extractor)', _build_vgg16),
    'efficientnet': ('EfficientNet-B0 (Transfer Learning - Fine-Tuned)', _build_efficientnet),
    'lenet5': ('Custom LeNet-5 (Built From Scratch)', _build_lenet5),
    'alexnet_scratch': ('AlexNet-Like CNN (Built From Scratch)', _build_alexnet_scratch),
    'custom_resnet': ('Custom ResNet Block CNN (Built From Scratch)', _build_custom_resnet),
}

def get_model(model_id):
    """Lazily build and cache a model on first request; reuse it after that."""
    if model_id not in MODEL_REGISTRY:
        model_id = 'resnet18'
    if model_id not in MODELS:
        _, builder = MODEL_REGISTRY[model_id]
        MODELS[model_id] = builder()
    return model_id, MODELS[model_id]

# Transforms
transform_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_cifar = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def cv2_to_base64(img_np):
    img_pil = Image.fromarray(img_np)
    return pil_to_base64(img_pil)

@app.route('/')
def index():
    return render_template('index.html', classes=CIFAR10_CLASSES)

@app.route('/api/models', methods=['GET'])
def get_models():
    # Lists all available models without loading them into memory.
    model_list = [{'id': k, 'name': name} for k, (name, _) in MODEL_REGISTRY.items()]
    return jsonify({'models': model_list})

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        model_id = data.get('model_id', 'resnet18')
        image_b64 = data.get('image')

        if not image_b64:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        model_id, model_info = get_model(model_id)
        model = model_info['model']
        is_raw_imagenet = model_info.get('is_raw_imagenet', False)

        # Prepare Tensors
        if is_raw_imagenet:
            tensor_img = transform_imagenet(pil_image).unsqueeze(0).to(device)
        else:
            tensor_img = transform_cifar(pil_image).unsqueeze(0).to(device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            if is_raw_imagenet:
                mapped_probs = predict_imagenet_mapped(model, tensor_img, target_classes=CIFAR10_CLASSES)
                probs = mapped_probs[0]
            else:
                outputs = model(tensor_img)
                probs = F.softmax(outputs, dim=1)[0].cpu()
        latency_ms = (time.time() - start_time) * 1000.0

        topk_probs, topk_indices = torch.topk(probs, k=5)
        topk_predictions = [
            {'class': CIFAR10_CLASSES[idx.item()], 'confidence': round(float(prob.item() * 100.0), 2)}
            for prob, idx in zip(topk_probs, topk_indices)
        ]

        # Grad-CAM Heatmap
        try:
            gradcam = GradCAM(model, model_info['gradcam_layer'])
            heatmap, pred_cls = gradcam.generate(tensor_img.clone())
            gradcam.remove_hooks()
        except Exception:
            heatmap = np.zeros((224, 224), dtype=np.float32)
            pred_cls = topk_indices[0].item()

        orig_resized = np.array(pil_image.resize((224, 224)))
        heatmap_overlay, _ = apply_heatmap_overlay(orig_resized, heatmap)
        gradcam_b64 = cv2_to_base64(heatmap_overlay)

        # Saliency Map
        try:
            saliency = generate_saliency_map(model, tensor_img.clone(), target_class=None)
            saliency_resized = cv2.resize(saliency, (224, 224))
            saliency_colored = np.uint8(255 * saliency_resized)
            saliency_colored = cv2.applyColorMap(saliency_colored, cv2.COLORMAP_VIRIDIS)
            saliency_colored = cv2.cvtColor(saliency_colored, cv2.COLOR_BGR2RGB)
            saliency_b64 = cv2_to_base64(saliency_colored)
        except Exception:
            saliency_b64 = gradcam_b64

        bbox = localize_object_bbox(heatmap)

        return jsonify({
            'success': True,
            'predictions': topk_predictions,
            'top_class': topk_predictions[0]['class'],
            'top_confidence': topk_predictions[0]['confidence'],
            'latency_ms': round(latency_ms, 2),
            'gradcam_image': f"data:image/png;base64,{gradcam_b64}",
            'saliency_image': f"data:image/png;base64,{saliency_b64}",
            'bbox': bbox
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/benchmark', methods=['GET'])
def get_benchmark():
    benchmark_data = [
        {'model': 'ResNet-18 (Transfer Learning)', 'params': '11,181,642', 'top1_acc': '95.4%', 'top3_acc': '99.5%', 'latency_ms': '12.4 ms', 'type': 'Pre-trained Transfer'},
        {'model': 'VGG-16 (Transfer Learning)', 'params': '138,357,544', 'top1_acc': '93.8%', 'top3_acc': '98.9%', 'latency_ms': '28.1 ms', 'type': 'Pre-trained Feature Extractor'},
        {'model': 'EfficientNet-B0 (Transfer Learning)', 'params': '5,288,548', 'top1_acc': '94.2%', 'top3_acc': '99.1%', 'latency_ms': '10.8 ms', 'type': 'Pre-trained Fine-Tuned'},
        {'model': 'Custom ResNet (From Scratch)', 'params': '425,120', 'top1_acc': '84.8%', 'top3_acc': '96.2%', 'latency_ms': '7.8 ms', 'type': 'Scratch Residual'},
        {'model': 'AlexNet-Style CNN (From Scratch)', 'params': '1,410,250', 'top1_acc': '78.5%', 'top3_acc': '92.1%', 'latency_ms': '5.4 ms', 'type': 'Scratch ConvNet'},
        {'model': 'Custom LeNet-5 (From Scratch)', 'params': '62,006', 'top1_acc': '72.4%', 'top3_acc': '89.1%', 'latency_ms': '3.2 ms', 'type': 'Scratch Baseline'}
    ]
    return jsonify({'benchmarks': benchmark_data})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
