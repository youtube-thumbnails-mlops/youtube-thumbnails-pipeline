import wandb
import os
import shutil

# Configuration matches train.py
PROJECT_NAME = "youtube-thumbnails-pipeline"
ENTITY = "daniele-acquaviva"
ARTIFACT_NAME = "thumbnail-classifier"
ALIAS = "prod"

def fetch_latest_model():
    """Download the latest model artifact from W&B."""
    print(f"🔄 Connecting to W&B to fetch {ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME}:{ALIAS}...")
    
    try:
        api = wandb.Api()
        artifact_path = f"{ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME}:{ALIAS}"
        artifact = api.artifact(artifact_path)
        
        # Download returns the directory containing the files
        print("⬇️  Downloading artifact...")
        artifact_dir = artifact.download()
        
        # We expect 'best_model.pth' inside
        model_path = os.path.join(artifact_dir, "best_model.pth")
        
        if os.path.exists(model_path):
            # Move/Rename to current directory as checkpoint.pth
            destination = "checkpoint.pth"
            shutil.move(model_path, destination)
            print(f"✅ Checkpoint downloaded and saved to {destination}")
            return True
        else:
            print(f"⚠️ Artifact found but 'best_model.pth' is missing in {artifact_dir}")
            return False
            
    except wandb.errors.CommError as e:
        print(f"ℹ️  Could not connect to W&B (Network issue?): {e}")
    except Exception as e:
        # Most likely Artifact not found (first run)
        print(f"ℹ️  No previous checkpoint found (First run?): {e}")
        
    return False

if __name__ == "__main__":
    fetch_latest_model()
