import wandb

# Config
ENTITY = "daniele5"
PROJECT = "youtube-thumbnails-pipeline"
REGISTRY = "model-registry"
MODEL_NAME = "thumbnail-classifier"

def cleanup():
    api = wandb.Api()
    
    print("🗑️  Starting Cleanup...")

    # 1. Delete from Registry
    # Project deletion via API is restricted/unavailable, so we clean the content.
    try:
        registry_collection = api.artifact_type("model", project=REGISTRY).collection(MODEL_NAME)
        print(f"   Found Registry Collection in '{REGISTRY}'. Deleting...")
        registry_collection.delete()
        print("   ✅ Registry cleaned.")
    except Exception as e:
        print(f"   ⚠️  Registry cleanup skipped (Not found?): {e}")

    # 2. Delete from Project (Warehouse)
    try:
        project_collection = api.artifact_type("model", project=PROJECT).collection(MODEL_NAME)
        print(f"   Found Project Collection. Deleting...")
        project_collection.delete()
        print("   ✅ Project artifacts cleaned.")
    except Exception as e:
        print(f"   ⚠️  Project artifact cleanup skipped: {e}")

    # 3. Delete ALL Runs (Experiments) in the Project
    # This removes the "Deleted-..." entries from the dashboard
    print(f"   Searching for Runs in {ENTITY}/{PROJECT}...")
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    if len(runs) > 0:
        print(f"   Found {len(runs)} Runs. Deleting them...")
        for run in runs:
            try:
                print(f"     - Deleting Run: {run.name} ({run.id})")
                run.delete()
            except Exception as e:
                print(f"       ⚠️ Failed to delete run {run.id}: {e}")
        print("   ✅ All runs deleted.")
    else:
        print("   ✅ No runs found.")

    print("✨ All clean!")

if __name__ == "__main__":
    cleanup()
