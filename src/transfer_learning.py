import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# ImageNet to 10 Target CIFAR-10 Categories mapping dictionary
IMAGENET_TO_CIFAR10_MAP = {
    'airplane': [404, 895, 466, 812, 405],
    'automobile': [817, 511, 436, 627, 751, 656, 705, 407, 468, 609, 661],
    'bird': list(range(7, 25)) + list(range(80, 101)),
    'cat': [281, 282, 283, 284, 285],
    'deer': [351, 352, 353],
    'dog': list(range(151, 269)),
    'frog': [30, 31, 32, 33],
    'horse': [339, 340],
    'ship': [510, 705, 724, 833, 472, 914, 814, 554, 625, 628, 871],
    'truck': [675, 867, 717, 569, 555, 864, 847, 734, 539]
}

CIFAR10_ORDER = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_pretrained_model(model_name='resnet18', num_classes=10, feature_extract=True, pretrained=True):
    """
    Loads a pre-trained model and adjusts its final classifier layer for custom classification.
    """
    model_name = model_name.lower()
    weights = 'DEFAULT' if pretrained else None

    if model_name == 'resnet18':
        model = models.resnet18(weights=weights)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_name == 'vgg16':
        model = models.vgg16(weights=weights)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif model_name in ['efficientnet_b0', 'efficientnet']:
        model = models.efficientnet_b0(weights=weights)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

def predict_imagenet_mapped(raw_imagenet_model, tensor_img, target_classes=CIFAR10_ORDER):
    """
    Takes an ImageNet pre-trained backbone, computes 1000-class probabilities,
    and aggregates probabilities mapped to the 10 target categories.
    """
    raw_imagenet_model.eval()
    with torch.no_grad():
        outputs = raw_imagenet_model(tensor_img)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    category_scores = []
    for cls_name in target_classes:
        indices = IMAGENET_TO_CIFAR10_MAP.get(cls_name, [])
        score = float(np.sum(probs[indices])) if indices else 0.0
        category_scores.append(score)

    category_scores = np.array(category_scores, dtype=np.float32)
    sum_score = np.sum(category_scores)
    if sum_score > 0:
        category_scores /= sum_score
    else:
        category_scores = np.ones(10, dtype=np.float32) / 10.0

    return torch.tensor(category_scores).unsqueeze(0)
