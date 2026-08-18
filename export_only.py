"""
Standalone export script.

Loads the ALREADY-TRAINED checkpoint saved by train_and_evaluate.py
(models/resnet18_transfer.pt) instead of retraining from scratch, then
runs ONNX / TorchScript / (optional) TFLite export.

Use this after a successful training run that failed only at the export
step (e.g. missing 'onnxscript' dependency) so you don't have to sit
through Phases 1-6 again.

Usage:
    python export_only.py
    python export_only.py --tflite
    python export_only.py --tflite --tflite-quantize
"""
import sys
import os
import argparse
import torch

sys.stdout.reconfigure(encoding='utf-8')

from src.transfer_learning import get_pretrained_model
from src.export import export_to_onnx, export_to_torchscript, export_to_tflite, export_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Export-only script (skips retraining)")
    parser.add_argument('--checkpoint', default='models/resnet18_transfer.pt',
                        help="Path to the trained model checkpoint (default: ResNet-18 transfer model).")
    parser.add_argument('--tflite', action='store_true',
                        help="Also export to TensorFlow Lite (requires: pip install -r requirements-tflite.txt).")
    parser.add_argument('--tflite-quantize', action='store_true',
                        help="Additionally produce an INT8 dynamic-range-quantized .tflite artifact.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at '{args.checkpoint}'.")
        print("Make sure you've already run train_and_evaluate.py at least through Phase 4 "
              "(ResNet-18 training), and check the 'models/' folder for the correct filename.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    resnet18 = get_pretrained_model(model_name='resnet18', num_classes=10,
                                     feature_extract=False, pretrained=False).to(device)
    resnet18.load_state_dict(torch.load(args.checkpoint, map_location=device))
    resnet18.eval()
    print("Checkpoint loaded successfully.")

    sample_tensor = torch.randn(1, 3, 32, 32).to(device)

    print("\n[Export] Exporting to ONNX...")
    export_to_onnx(resnet18, sample_tensor.cpu(), output_path='models/resnet18.onnx')

    print("\n[Export] Exporting to TorchScript...")
    export_to_torchscript(resnet18, sample_tensor.cpu(), output_path='models/resnet18_jit.pt')

    tflite_exported = False
    if args.tflite:
        print("\n[Export] Exporting to TensorFlow Lite...")
        try:
            export_to_tflite(
                resnet18, sample_tensor.cpu(),
                onnx_path='models/resnet18.onnx',
                tflite_output_dir='models/tflite_export',
                quantize=args.tflite_quantize,
            )
            tflite_exported = True
        except ImportError as e:
            print(f"[Export] Skipped TFLite export -- missing optional dependencies.\n{e}")

    export_summary(resnet18, save_path='models/export_summary.txt', tflite_exported=tflite_exported)

    print("\n✅ Export completed successfully! Check the 'models/' folder for output files.")


if __name__ == '__main__':
    main()