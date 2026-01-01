import wandb
import os

PROJECT_NAME = "youtube-thumbnails-pipeline"
ARTIFACT_TYPE = "model"
ARTIFACT_NAME = "thumbnail-classifier"

def verify():
    # Check the REGISTRY project
    REGISTRY_PRJ = "model-registry"
    print(f"🔍 Checking Project '{REGISTRY_PRJ}' for promoted models...")
    
    api = wandb.Api()
    try:
        collections = api.artifact_type("model", project=REGISTRY_PRJ).collections()
        for col in collections:
            print(f"\n📚 Collection: {col.name}")
            for v in col.artifacts():
                print(f"   🔹 Version: {v.version} (Aliases: {v.aliases})")
    except Exception as e:
        print(f"❌ Registry Check Failed: {e}")

if __name__ == "__main__":
    verify()
