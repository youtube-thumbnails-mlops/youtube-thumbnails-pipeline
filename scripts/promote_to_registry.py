import wandb
import argparse
import sys

# Configuration
PROJECT_NAME = "youtube-thumbnails-pipeline"
ENTITY = "daniele5"  # Hardcoded based on previous logs, could be env var
MODEL_NAME = "thumbnail-classifier"
REGISTRY_PROJECT = "model-registry" # This backs the official W&B Registry UI

def promote(version_alias="latest", target_alias="production"):
    print(f"🚀 Promoting '{MODEL_NAME}:{version_alias}' to Registry as '{target_alias}'...")
    
    api = wandb.Api()
    
    # 1. Fetch the Source Artifact
    artifact_path = f"{ENTITY}/{PROJECT_NAME}/{MODEL_NAME}:{version_alias}"
    try:
        artifact = api.artifact(artifact_path)
        print(f"✅ Found source artifact: {artifact.name} (ID: {artifact.id})")
    except wandb.errors.CommError:
        print(f"❌ Error: Could not find artifact at {artifact_path}")
        sys.exit(1)

    # 2. Define Target Registry Path
    # User provided explicit path: daniele5/model-registry/thumbnail-classifier
    REGISTRY_PROJECT = "model-registry"
    target_path = f"{ENTITY}/{REGISTRY_PROJECT}/{MODEL_NAME}"
    
    # --- CHAMPION / CHALLENGER LOGIC ---
    try:
        # Try to fetch current production model
        prod_artifact = api.artifact(f"{target_path}:production")
        prod_acc = prod_artifact.metadata.get("val_acc", 0.0)
        print(f"🥊 Current Champion (Production) Accuracy: {prod_acc:.4f}")
    except (wandb.errors.CommError, KeyError):
        print("🆕 No production model found (or no metadata). This is the first one.")
        prod_acc = -1.0

    # Get candidate accuracy
    cand_acc = artifact.metadata.get("val_acc", 0.0)
    print(f"🥊 Candidate Challenger Accuracy: {cand_acc:.4f}")

    if cand_acc >= prod_acc:
        print(f"✅ Challenger ({cand_acc:.4f}) beats/ties Champion ({prod_acc:.4f}). PROMOTING! 🚀")
        
        # 3. Link and Alias
        try:
            artifact.link(target_path, aliases=[target_alias])
            print(f"✅ Successfully promoted to Production!")
            print(f"🏆 Registry Link: https://wandb.ai/{ENTITY}/{REGISTRY_PROJECT}/artifacts/model/{MODEL_NAME}")
        except Exception as e:
            print(f"❌ Promotion Link Failed: {e}")
            sys.exit(1)
    else:
        print(f"🛑 Challenger ({cand_acc:.4f}) is WORSE than Champion ({prod_acc:.4f}). NOT Promoting.")
        # We don't exit(1) because training was successful, just not good enough for prod.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote a model artifact to Production in W&B Registry.")
    parser.add_argument("--version", type=str, default="latest", help="Version tag/alias to promote (default: latest)")
    parser.add_argument("--alias", type=str, default="production", help="Target alias in registry (default: production)")
    
    args = parser.parse_args()
    promote(args.version, args.alias)
