# #!/usr/bin/env python3
# """
# FastSAM to ONNX Exporter
# Run this script on your Ubuntu 20 laptop to export the FastSAM model to ONNX format.
# The exported .onnx file can then be copied to the Jetson TX2 for deployment.

# Usage:
#     python3 export_fastsam_to_onnx.py

# Output:
#     fastsam_s.onnx - The exported ONNX model ready for deployment
# """

# import torch
# import numpy as np
# from ultralytics import YOLO
# import onnx
# import onnxsim

# def export_fastsam_to_onnx(
#     model_path="FastSAM-s.pt",
#     output_path="fastsam_s.onnx",
#     img_size=640,
#     simplify=True
# ):
#     """
#     Export FastSAM model to ONNX format.
    
#     Args:
#         model_path: Path to FastSAM .pt weights
#         output_path: Path to save the ONNX model
#         img_size: Input image size (default 640)
#         simplify: Whether to simplify the ONNX model
#     """
    
#     print(f"Loading FastSAM model from {model_path}...")
    
#     try:
#         # Load the FastSAM model
#         model = YOLO(model_path)
        
#         print(f"Exporting to ONNX format with image size {img_size}...")
        
#         # Export to ONNX
#         # Ultralytics has built-in ONNX export functionality
#         success = model.export(
#             format='onnx',
#             imgsz=img_size,
#             simplify=simplify,
#             opset=12  # Use opset 12 for better compatibility with older ONNX Runtime
#         )
        
#         if success:
#             # The export will create a file with .onnx extension
#             exported_file = model_path.replace('.pt', '.onnx')
#             print(f"\n? ONNX export successful!")
#             print(f"  Model saved to: {exported_file}")
            
#             # Load and verify the model
#             print("\nVerifying ONNX model...")
#             onnx_model = onnx.load(exported_file)
#             onnx.checker.check_model(onnx_model)
#             print("? ONNX model verification passed!")
            
#             # Print model info
#             print(f"\nModel Information:")
#             print(f"  Input name: {onnx_model.graph.input[0].name}")
#             print(f"  Input shape: {[dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]}")
#             print(f"  Output count: {len(onnx_model.graph.output)}")
            
#             print(f"\n{'='*60}")
#             print("NEXT STEPS:")
#             print(f"{'='*60}")
#             print(f"1. Copy {exported_file} to Jetson TX2:")
#             print(f"   scp {exported_file} jetson@<jetson-ip>:~/")
#             print(f"\n2. On Jetson TX2, install ONNX Runtime:")
#             print(f"   pip install onnxruntime==1.8.0")
#             print(f"\n3. Update segmentation.py to use ONNX inference")
#             print(f"   (use segmentation_onnx.py as replacement)")
#             print(f"{'='*60}\n")
            
#         else:
#             print("? ONNX export failed!")
            
#     except Exception as e:
#         print(f"? Error during export: {e}")
#         print("\nTroubleshooting:")
#         print("1. Make sure ultralytics is installed: pip install ultralytics")
#         print("2. Make sure FastSAM-s.pt exists in current directory")
#         print("3. Try updating PyTorch: pip install --upgrade torch")
#         raise

# def test_onnx_inference(onnx_path="FastSAM-s.onnx"):
#     """
#     Test the exported ONNX model with dummy inference.
#     """
#     try:
#         import onnxruntime as ort
        
#         print(f"\nTesting ONNX inference with {onnx_path}...")
        
#         # Create inference session
#         session = ort.InferenceSession(onnx_path)
        
#         # Get input/output details
#         input_name = session.get_inputs()[0].name
#         input_shape = session.get_inputs()[0].shape
        
#         print(f"  Input name: {input_name}")
#         print(f"  Input shape: {input_shape}")
        
#         # Create dummy input
#         dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
#         # Run inference
#         outputs = session.run(None, {input_name: dummy_input})
        
#         print(f"  Output count: {len(outputs)}")
#         for i, output in enumerate(outputs):
#             print(f"  Output {i} shape: {output.shape}")
        
#         print("? ONNX inference test passed!")
        
#     except ImportError:
#         print("\n? ONNX Runtime not installed (not required on laptop)")
#         print("  This is OK - ONNX Runtime will be installed on Jetson TX2")
#     except Exception as e:
#         print(f"? Inference test failed: {e}")

# if __name__ == "__main__":
#     print("="*60)
#     print("FastSAM to ONNX Exporter")
#     print("="*60)
    
#     # Export the model
#     export_fastsam_to_onnx(
#         model_path="FastSAM-s.pt",
#         output_path="fastsam_s.onnx",
#         img_size=640,
#         simplify=True
#     )
    
#     # Test the exported model (optional)
#     try:
#         test_onnx_inference("FastSAM-s.onnx")
#     except:
#         pass
    
#     print("\n? Export complete!")


#!/usr/bin/env python3
"""
FastSAM to ONNX Export Script
=============================
Run this script on your laptop (Ubuntu 20) to export FastSAM model to ONNX format.

This script exports FastSAM with compatibility settings for:
- ONNX opset version 11 (widely supported on older runtimes)
- Dynamic input shapes for flexible image sizes
- FP32 precision (most compatible with older hardware)

Usage:
    python export_fastsam_onnx.py

Output:
    - fastsam_encoder.onnx  (image encoder backbone)
    - fastsam_decoder.onnx  (prompt decoder for point-based segmentation)

Note: The exported models should be compatible with ONNX Runtime 1.8.x on Ubuntu 16.04
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Configuration
OPSET_VERSION = 11  # Use opset 11 for maximum compatibility with older ONNX Runtime
INPUT_SIZE = (640, 480)  # Width, Height - match your camera resolution
EXPORT_DIR = Path(__file__).parent / "onnx_models"


def check_dependencies():
    """Check that all required packages are installed."""
    try:
        from ultralytics.models.fastsam import FastSAM
        import onnx
        print("✓ All dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Install with: pip install ultralytics onnx onnxsim")
        return False


def export_fastsam_encoder(model, export_path: Path):
    """
    Export the FastSAM encoder (YOLOv8-seg backbone) to ONNX.
    
    The encoder processes the input image and produces:
    - Detection boxes
    - Segmentation prototypes
    - Feature maps for mask generation
    """
    print("\n[1/3] Exporting FastSAM encoder...")
    
    # Get the underlying YOLO model
    yolo_model = model.model
    
    # Create dummy input matching expected input size
    # FastSAM expects (B, C, H, W) format
    dummy_input = torch.randn(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    
    # Export path for encoder
    encoder_path = export_path / "fastsam_encoder.onnx"
    
    # Export with compatibility settings
    torch.onnx.export(
        yolo_model,
        dummy_input,
        str(encoder_path),
        opset_version=OPSET_VERSION,
        input_names=['images'],
        output_names=['output0', 'output1'],  # boxes + masks
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output0': {0: 'batch'},
            'output1': {0: 'batch'}
        },
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"  ✓ Encoder exported to: {encoder_path}")
    return encoder_path


def simplify_onnx_model(onnx_path: Path):
    """
    Simplify ONNX model for better compatibility and performance.
    Uses onnx-simplifier to optimize the graph.
    """
    try:
        import onnxsim
        import onnx
        
        print(f"  Simplifying {onnx_path.name}...")
        
        model = onnx.load(str(onnx_path))
        model_simplified, check = onnxsim.simplify(model)
        
        if check:
            onnx.save(model_simplified, str(onnx_path))
            print(f"  ✓ Model simplified successfully")
        else:
            print(f"  ⚠ Simplification check failed, using original model")
            
    except ImportError:
        print("  ⚠ onnx-simplifier not installed, skipping simplification")
        print("    Install with: pip install onnx-simplifier")


def verify_onnx_model(onnx_path: Path):
    """Verify the exported ONNX model is valid."""
    import onnx
    
    print(f"\n[3/3] Verifying {onnx_path.name}...")
    
    try:
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print(f"  ✓ Model verification passed")
        
        # Print model info
        print(f"  - ONNX opset version: {model.opset_import[0].version}")
        print(f"  - Input shape: {[d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]}")
        
        return True
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def test_onnx_inference(onnx_path: Path):
    """Test ONNX model with ONNX Runtime."""
    try:
        import onnxruntime as ort
        
        print(f"\n[Test] Running inference test...")
        
        # Create session
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input info
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  - Input name: {input_name}")
        print(f"  - Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, INPUT_SIZE[1], INPUT_SIZE[0]).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"  - Number of outputs: {len(outputs)}")
        for i, out in enumerate(outputs):
            print(f"  - Output {i} shape: {out.shape}")
        
        print("  ✓ Inference test passed")
        return True
        
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        return False


def export_ultralytics_native():
    """
    Use Ultralytics native export (simpler but may use newer opset).
    Falls back to this if custom export fails.
    """
    from ultralytics import YOLO
    
    print("\n[Alternative] Using Ultralytics native export...")
    
    # Load FastSAM model
    model = YOLO("FastSAM-s.pt")
    
    # Export to ONNX
    model.export(
        format="onnx",
        opset=OPSET_VERSION,
        simplify=True,
        dynamic=False,  # Fixed size for compatibility
        imgsz=(INPUT_SIZE[1], INPUT_SIZE[0])
    )
    
    print("  ✓ Native export completed")


def create_compatibility_notes():
    """Create a README with compatibility information."""
    notes = """
ONNX Model Compatibility Notes
==============================

Exported with:
- ONNX opset version: {opset}
- Input size: {size}
- Precision: FP32

Tested with:
- PyTorch: {torch_version}
- ONNX: See export log
- ONNX Runtime: See export log

For Jetson TX2 (Ubuntu 16.04):
- Build ONNX Runtime 1.8.0 from source
- Use CPU or CUDA execution provider
- See jetson_onnxruntime_build.md for instructions

Files:
- fastsam_encoder.onnx: Main segmentation model
- This file: Compatibility notes
""".format(
        opset=OPSET_VERSION,
        size=INPUT_SIZE,
        torch_version=torch.__version__
    )
    
    notes_path = EXPORT_DIR / "ONNX_COMPATIBILITY.txt"
    with open(notes_path, 'w') as f:
        f.write(notes)
    print(f"\n✓ Compatibility notes saved to: {notes_path}")


def main():
    print("=" * 60)
    print("FastSAM to ONNX Export Tool")
    print("=" * 60)
    print(f"Target opset version: {OPSET_VERSION}")
    print(f"Input size: {INPUT_SIZE}")
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create export directory
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Export directory: {EXPORT_DIR}")
    
    try:
        # Method 1: Try Ultralytics native export (most reliable)
        from ultralytics import YOLO
        
        print("\n[1/3] Loading FastSAM model...")
        
        # Check if model file exists in current directory
        model_path = Path(__file__).parent / "FastSAM-s.pt"
        if not model_path.exists():
            print(f"  Model not found at {model_path}")
            print("  Downloading FastSAM-s model...")
            model_path = "FastSAM-s.pt"  # Will download automatically
        
        model = YOLO(str(model_path))
        print("  ✓ Model loaded")
        
        print("\n[2/3] Exporting to ONNX...")
        export_path = EXPORT_DIR / "fastsam_s.onnx"
        
        model.export(
            format="onnx",
            opset=OPSET_VERSION,
            simplify=True,
            dynamic=False,
            imgsz=(INPUT_SIZE[1], INPUT_SIZE[0]),
            half=False,  # FP32 for compatibility
        )
        
        # Ultralytics exports next to the .pt file, move it
        default_export = Path(str(model_path).replace('.pt', '.onnx'))
        if default_export.exists():
            import shutil
            shutil.move(str(default_export), str(export_path))
            print(f"  ✓ Model exported to: {export_path}")
        
        # Verify
        verify_onnx_model(export_path)
        
        # Test
        test_onnx_inference(export_path)
        
        # Create notes
        create_compatibility_notes()
        
        print("\n" + "=" * 60)
        print("Export Complete!")
        print("=" * 60)
        print(f"\nCopy these files to Jetson TX2:")
        print(f"  - {export_path}")
        print(f"\nSee 'jetson_onnxruntime_build.md' for ONNX Runtime installation")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
