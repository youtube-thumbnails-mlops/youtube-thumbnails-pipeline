import os
import wandb
import sys
import torch
import torch.nn as nn
from torchvision import models

# Config
PROJECT_NAME = "youtube-thumbnails-pipeline"
CATEGORY_MAP = {
    1: 0, 2: 1, 10: 2, 15: 3, 17: 4, 19: 5, 20: 6, 22: 7, 23: 8, 24: 9,
    25: 10, 26: 11, 27: 12, 28: 13, 29: 14, 30: 15, 43: 16
}
NUM_CLASSES = len(CATEGORY_MAP)

def deploy():
    print("🚀 Initializing Initial Model Deployment (ResNet18)...")
    
    run = wandb.init(
        project=PROJECT_NAME, 
        job_type="initialize-model",
        tags=["v0", "initialization", "resnet18"]
    )
    
    print("📥 Downloading ResNet18 ImageNet weights...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify last layer for our N classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.eval() 
    
    print("📦 Exporting Weights & ONNX...")
    
    pth_filename = "best_model.pth"
    torch.save(model.state_dict(), pth_filename)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_filename = "model.onnx"
    
    # Debug info
    param_size = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Parameters: {param_size/1e6:.2f} M")
    print(f"🔧 PyTorch: {torch.__version__} (Should be 2.6.0)")

    try:
        # Standard export for PyTorch 2.6.0
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_filename, 
            verbose=False, 
            export_params=True,
            opset_version=17, # Standard for 2.x
            input_names=['input'], 
            output_names=['output']
        )
    except Exception as e:
        print(f"❌ Export Failed: {e}")
        raise

    if not os.path.exists(onnx_filename):
        raise FileNotFoundError(f"❌ ONNX file {onnx_filename} was not created!")
        
    # Verification
    size_mb = os.path.getsize(onnx_filename) / (1024 * 1024)
    print(f"🔍 Generated ONNX Size: {size_mb:.2f} MB")
    
    # ResNet18 should be ~44 MB
    if size_mb < 40:
        raise ValueError(f"❌ ONNX file is suspicious ({size_mb:.2f} MB). Weights missing?")
        
    print("🧪 Verifying model with ONNX Runtime...")
    try:
        import onnxruntime as ort
        import numpy as np
        ort_session = ort.InferenceSession(onnx_filename)
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        ort_outs = ort_session.run(None, ort_inputs)
        print(f"✅ Inference Check Passed! Output shape: {ort_outs[0].shape}")
    except ImportError:
        print("⚠️ 'onnxruntime' not found. Skipping check.")
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        raise
    
    print("⬆️  Uploading to W&B Registry...")
    artifact = wandb.Artifact(
        'thumbnail-classifier', 
        type='model',
        description="Initial v0 model (ResNet18)",
        metadata={"num_classes": NUM_CLASSES, "base": "resnet18"}
    )
    artifact.add_file(pth_filename)
    artifact.add_file(onnx_filename)
    
    run.log_artifact(artifact, aliases=["latest"])
    
    os.remove(onnx_filename)
    os.remove(pth_filename)
    
    wandb.finish()
    print(f"✅ Deployment Complete!")
    
    print("\n🚀 Triggering Registry Promotion...")
    import subprocess
    promote_script = os.path.join(os.path.dirname(__file__), "promote_to_registry.py")
    try:
        # Pass the same python executable (venv_stable)
        subprocess.run([sys.executable, promote_script, "--version", "latest", "--alias", "production"], check=True)
        print("✅ Automatic Promotion Successful!")
    except subprocess.CalledProcessError:
        print("⚠️ Automatic Promotion Failed.")

if __name__ == "__main__":
    deploy()
