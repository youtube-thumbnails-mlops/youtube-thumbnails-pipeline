import os
import wandb
import sys
import torch
import torch.nn as nn
from torchvision import models

# Config
PROJECT_NAME = "youtube-thumbnails-pipeline"
# Categories from collection/train script
CATEGORY_MAP = {
    1: 0, 2: 1, 10: 2, 15: 3, 17: 4, 19: 5, 20: 6, 22: 7, 23: 8, 24: 9,
    25: 10, 26: 11, 27: 12, 28: 13, 29: 14, 30: 15, 43: 16
}
NUM_CLASSES = len(CATEGORY_MAP)

def deploy():
    print("🚀 Initializing Initial Model Deployment...")
    
    # 1. Init W&B
    # We use a specific job_type to distinguish this from training runs
    run = wandb.init(
        project=PROJECT_NAME, 
        job_type="initialize-model",
        tags=["v0", "initialization", "resnet18"]
    )
    
    # 2. Setup Model (Standard ResNet18)
    print("📥 Downloading ImageNet weights...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify last layer for our N classes (random weights here)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.eval() # Set to eval mode for export
    
    # 3. Export to ONNX & Save Checkpoint
    print("📦 Exporting Weights & ONNX...")
    
    # Save .pth for training script (finetuning)
    pth_filename = "best_model.pth"
    torch.save(model.state_dict(), pth_filename)
    
    # Save .onnx for website
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_filename = "model.onnx"
    
    # Use Opset 18 (Required by this PyTorch version)
    # Remove dynamic_axes for stability with newer PyTorch exporters
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_filename, 
            verbose=False,
            opset_version=18,
            input_names=['input'],
            output_names=['output']
        )
    except Exception as e:
        print(f"❌ Export Failed: {e}")
        raise

    if not os.path.exists(onnx_filename):
        raise FileNotFoundError(f"❌ ONNX file {onnx_filename} was not created!")
    
    # 4. Log to W&B Registry
    print("⬆️  Uploading to Weights & Biases Registry...")
    artifact = wandb.Artifact(
        'thumbnail-classifier', 
        type='model',
        description="Initial v0 model (ResNet18 ImageNet weights, un-finetuned)",
        metadata={"num_classes": NUM_CLASSES, "base": "resnet18"}
    )
    # Add BOTH files: one for training, one for inference
    artifact.add_file(pth_filename)
    artifact.add_file(onnx_filename)
    
    run.log_artifact(artifact, aliases=["latest"])
    
    # Clean up local file
    os.remove(onnx_filename)
    os.remove(pth_filename)
    # os.remove("version_tag.txt") # Removed test hack
    
    wandb.finish()
    print(f"✅ Deployment Complete! Model artifact is ready.")
    
    # 5. Automate Promotion to Registry
    print("\n🚀 Triggering Registry Promotion...")
    import subprocess
    promote_script = os.path.join(os.path.dirname(__file__), "promote_to_registry.py")
    try:
        # Use 'latest' to capture the version we just uploaded
        subprocess.run([sys.executable, promote_script, "--version", "latest", "--alias", "production"], check=True)
        print("✅ Automatic Promotion Successful!")
    except subprocess.CalledProcessError:
        print("⚠️ Automatic Promotion Failed (Network issue?). You can run 'scripts/promote_to_registry.py' manually.")

if __name__ == "__main__":
    deploy()
