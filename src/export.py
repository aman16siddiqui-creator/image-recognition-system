import os
import torch

def export_to_onnx(model, dummy_input, output_path='models/model.onnx', input_names=['input'], output_names=['output']):
    """
    Exports a PyTorch model to ONNX format.
    """
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model successfully exported to ONNX format at: {output_path}")
    return output_path


def export_to_torchscript(model, dummy_input, output_path='models/model_jit.pt'):
    """
    Exports a PyTorch model to TorchScript JIT format.
    """
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(output_path)
    print(f"Model successfully exported to TorchScript at: {output_path}")
    return output_path


def export_summary(model, save_path='models/export_summary.txt'):
    """
    Saves export metadata and architecture summary.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("=== Model Deployment & Export Summary ===\n")
        f.write(f"Architecture: {model.__class__.__name__}\n")
        total_params = sum(p.numel() for p in model.parameters())
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write("Exports Generated: PyTorch .pt, ONNX .onnx, TorchScript .pt\n")
    return save_path
