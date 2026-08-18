import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM implementation to compute Class Activation Maps for target convolutional layers.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate(self, input_image, target_class=None):
        """
        Generates Grad-CAM heatmap for an input image tensor (1, C, H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        input_image.requires_grad = True
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()      # (C, H, W)
        activations = self.activations[0].cpu().data.numpy()  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))              # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize CAM to match input image height and width
        H, W = input_image.shape[2:]
        cam = cv2.resize(cam, (W, H))
        return cam, target_class


def apply_heatmap_overlay(original_img_np, heatmap, alpha=0.5):
    """
    Overlays Grad-CAM heatmap onto RGB image [0-255].
    """
    # Ensure original_img_np is uint8
    if original_img_np.dtype != np.uint8:
        if original_img_np.max() <= 1.0:
            original_img_np = np.uint8(255 * original_img_np)
        else:
            original_img_np = np.uint8(original_img_np)

    # Ensure 3 channels (RGB)
    if original_img_np.ndim == 2:
        original_img_np = cv2.cvtColor(original_img_np, cv2.COLOR_GRAY2RGB)
    elif original_img_np.ndim == 3 and original_img_np.shape[2] == 4:
        original_img_np = original_img_np[:, :, :3]

    H, W = original_img_np.shape[:2]

    # Resize heatmap to match original image dimensions exactly
    heatmap_resized = cv2.resize(heatmap, (W, H))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay, heatmap_colored


def extract_feature_maps(model, input_image, max_maps=16):
    """
    Extracts intermediate feature activations across all Convolutional layers.
    """
    model.eval()
    conv_maps = {}

    def get_hook(layer_name):
        def hook(module, input, output):
            conv_maps[layer_name] = output.detach().cpu()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        _ = model(input_image)

    for h in hooks:
        h.remove()

    processed_maps = {}
    for layer_name, fmaps in conv_maps.items():
        # Shape: (1, C, H, W) -> take first max_maps channels
        maps_np = fmaps[0].numpy()
        num_channels = min(maps_np.shape[0], max_maps)
        channel_imgs = []
        for i in range(num_channels):
            c_map = maps_np[i]
            c_map = (c_map - c_map.min()) / (c_map.max() - c_map.min() + 1e-8)
            channel_imgs.append(np.uint8(255 * c_map))
        processed_maps[layer_name] = channel_imgs

    return processed_maps


def generate_saliency_map(model, input_image, target_class=None):
    """
    Generates vanilla gradient Saliency Map for input image.
    """
    model.eval()
    input_image = input_image.clone().detach()
    input_image.requires_grad = True

    output = model(input_image)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[0, target_class]
    score.backward()

    saliency, _ = torch.max(input_image.grad.data.abs(), dim=1)
    saliency_np = saliency[0].cpu().numpy()
    saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)
    return saliency_np


def localize_object_bbox(heatmap, threshold=0.5):
    """
    Extracts estimated bounding box (x, y, w, h) from Grad-CAM activation threshold.
    """
    binary_mask = np.uint8(heatmap > threshold) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
