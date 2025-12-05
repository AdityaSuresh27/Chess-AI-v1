"""
PyTorch to ONNX Converter for Chess AI - WEB OPTIMIZED
Creates a single-file ONNX model 

I used this for my 3d website portfolio project(ONNX=No Server=Static Webpage=Able to host on github) 
"""

import torch
import torch.nn as nn
import numpy as np
import os

class AdvancedChessNet(nn.Module):
    """
    Multi-task network - MUST match your training code exactly
    """
    def __init__(self, input_size=791, output_size=4096):
        super(AdvancedChessNet, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
        )
        
        # Move prediction head
        self.move_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_size)
        )
        
        # Position evaluation head
        self.eval_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        shared_features = self.shared(x)
        move_logits = self.move_head(shared_features)
        eval_score = self.eval_head(shared_features)
        return move_logits, eval_score

def convert_to_onnx(
    pth_path='chess_model_advanced.pth',
    onnx_path='chess_model.onnx',
    input_size=791,
    output_size=4096
):
    
    print("="*70)
    print("PYTORCH TO ONNX CONVERTER")
    print("="*70)
    
    # Check if model file exists
    if not os.path.exists(pth_path):
        print(f"\n❌ ERROR: Model file not found: {pth_path}")
        print("\nMake sure you:")
        print("1. Created a separate folder for conversion")
        print("2. Copied chess_model_advanced.pth to this folder")
        print("3. Running this script from that folder")
        return False
    
    print(f"\n✓ Found model file: {pth_path}")
    file_size_mb = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.2f} MB")
    
    print(f"\n Loading PyTorch model...")
    device = torch.device('cpu')
    model = AdvancedChessNet(input_size=input_size, output_size=output_size)
    
    try:
        checkpoint = torch.load(pth_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch'] + 1}")
            if 'val_acc' in checkpoint:
                print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        
        print("✓ Model loaded successfully!")
        
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("\nMake sure the model architecture matches your training code!")
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input
    print(f"\n Creating dummy input...")
    dummy_input = torch.randn(1, input_size)
    print(f"  Input shape: {dummy_input.shape}")
    
    # Test the model works
    print(f"\nTesting model inference...")
    try:
        with torch.no_grad():
            move_logits, eval_score = model(dummy_input)
        print(f"  Move logits shape: {move_logits.shape}")
        print(f"  Eval score shape: {eval_score.shape}")
        print("✓ Model inference successful!")
    except Exception as e:
        print(f"\nERROR during inference: {e}")
        return False
    
    # Export to ONNX (single file - no external data)
    print(f"\nConverting to ONNX format (single file)...")
    print(f"  Output file: {onnx_path}")
    print(f"  Using opset_version=11 (IR v6 - compatible with ONNX Runtime Web)")
    print(f"  All data embedded (no .data file)")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Use opset 11 for better compatibility
            do_constant_folding=True,
            input_names=['board_features'],
            output_names=['move_logits', 'eval_score'],
            dynamic_axes={
                'board_features': {0: 'batch_size'},
                'move_logits': {0: 'batch_size'},
                'eval_score': {0: 'batch_size'}
            }
        )
        print("✓ ONNX export successful!")
        
    except Exception as e:
        print(f"\nERROR during ONNX export: {e}")
        return False
    
    # CRITICAL: Remove external data file if it was created
    data_file = onnx_path + '.data'
    if os.path.exists(data_file):
        print(f"\nExternal data file detected: {data_file}")
        print(f"   This will cause issues in the browser!")
        print(f"   We need to merge it into a single file...")
        
        try:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data, convert_model_from_external_data
            
            # Load model and convert from external data to internal
            print(f"   Loading ONNX model...")
            onnx_model = onnx.load(onnx_path)
            
            print(f"   Converting to single file...")
            # This converts external data back into the model
            convert_model_from_external_data(onnx_model)
            
            # Save as single file
            onnx.save(onnx_model, onnx_path)
            
            # Remove the .data file
            os.remove(data_file)
            print(f"   ✓ Merged into single file!")
            
        except Exception as e:
            print(f"    Failed to merge: {e}")
            print(f"   You may need to install: pip install onnx")
            return False
    
    # Verify ONNX model
    print(f"\nVerifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")
        
        # Check if external data exists (it shouldn't!)
        has_external = any(
            hasattr(tensor, 'external_data') and len(tensor.external_data) > 0
            for tensor in onnx_model.graph.initializer
        )
        
        if has_external:
            print("WARNING: Model still has external data references!")
            print("   This will NOT work in the browser.")
            return False
        else:
            print("✓ All data embedded in single file (no external .data file)")
        
        onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\nFile Size:")
        print(f"  PyTorch (.pth): {file_size_mb:.2f} MB")
        print(f"  ONNX (.onnx):   {onnx_size_mb:.2f} MB")
        
        if onnx_size_mb > 100:
            print(f"\nWarning: Model is {onnx_size_mb:.1f} MB")
            print(f"   This may be slow to load in browsers.")
            print(f"   Consider using a smaller model architecture.")
        
    except ImportError:
        print("Warning: 'onnx' package not installed. Skipping verification.")
        print("  Install with: pip install onnx")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")
    
    # Test ONNX inference
    print(f"\nTesting ONNX inference...")
    try:
        import onnxruntime as ort
        
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {'board_features': dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✓ ONNX Runtime inference successful!")
        print(f"  Output 1 (move_logits) shape: {ort_outputs[0].shape}")
        print(f"  Output 2 (eval_score) shape: {ort_outputs[1].shape}")
        
        # Compare with PyTorch output
        with torch.no_grad():
            pt_move, pt_eval = model(dummy_input)
        
        move_diff = np.abs(pt_move.numpy() - ort_outputs[0]).max()
        eval_diff = np.abs(pt_eval.numpy() - ort_outputs[1]).max()
        
        print(f"\n📊 PyTorch vs ONNX comparison:")
        print(f"  Max difference in move_logits: {move_diff:.6f}")
        print(f"  Max difference in eval_score:  {eval_diff:.6f}")
        
        if move_diff < 1e-4 and eval_diff < 1e-4:
            print("  ✓ Outputs match!")
        else:
            print("  ⚠ Small differences (normal)")
        
    except ImportError:
        print("⚠ Warning: 'onnxruntime' not installed. Skipping test.")
        print("  Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠ Warning: ONNX inference test failed: {e}")
    
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print(f"Test in your browser(if you want)")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    PTH_FILE = 'chess_model_advanced.pth'
    ONNX_FILE = 'chess_model.onnx'
    
    print("Configuration:")
    print(f"  Input:  {PTH_FILE}")
    print(f"  Output: {ONNX_FILE} (single file)")
    print(f"  Input features:  791")
    print(f"  Output classes:  4096")
    print()
    

    print("Checking dependencies...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  PyTorch not found! Install: pip install torch")
        exit(1)
    
    try:
        import onnx
        print(f"  ✓ ONNX {onnx.__version__}")
    except ImportError:
        print("  ONNX not found (recommended). Install: pip install onnx")
    
    try:
        import onnxruntime
        print(f"  ✓ ONNX Runtime {onnxruntime.__version__}")
    except ImportError:
        print("  ⚠ ONNX Runtime not found (optional). Install: pip install onnxruntime")
    
    print()
    
    # Run conversion
    success = convert_to_onnx(
        pth_path=PTH_FILE,
        onnx_path=ONNX_FILE,
        input_size=791,
        output_size=4096
    )
    
    if success:
        print("All done! Your model is ready for the browser!")
        print("\nPro tip: Test locally first with a simple web server:")
        print("   python -m http.server 8000")
        print("   Then open: http://localhost:8000")
    else:
        print("Conversion failed. Please check the errors above.")
        exit(1)