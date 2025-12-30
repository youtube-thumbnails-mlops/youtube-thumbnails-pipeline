import os
import sys
import wandb
import requests
from dotenv import load_dotenv

# Add parent dir to path to import submit_job
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from submit_job import submit_job

load_dotenv()

PROJECT_NAME = "youtube-thumbnails-train"
DATASET_REPO = "daniele-acquaviva/youtube-thumbnails-dataset" # Adjust user/org
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_latest_trained_version():
    """Get the batch_version of the last successful run from W&B."""
    try:
        api = wandb.Api()
        runs = api.runs(f"daniele-acquaviva/{PROJECT_NAME}", order="-created_at")
        for run in runs:
            if run.state == "finished" and "batch_version" in run.config:
                return run.config["batch_version"]
    except Exception as e:
        print(f"⚠️ Could not fetch runs from W&B: {e}")
    return "batch_000"

def get_latest_dataset_batch():
    """Get the latest batch folder from the dataset repo via GitHub API."""
    url = f"https://api.github.com/repos/{DATASET_REPO}/contents/batches"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"⚠️ GitHub API Error: {resp.status_code}")
            return "batch_000"
            
        data = resp.json()
        batches = [item['name'] for item in data if item['name'].startswith('batch_')]
        if not batches:
            return "batch_000"
            
        # Sort simple strings batch_001, batch_002...
        return sorted(batches)[-1]
    except Exception as e:
        print(f"⚠️ Could not fetch dataset batches: {e}")
        return "batch_000"

def main():
    print("🔍 Checking for new data...")
    
    last_trained = get_latest_trained_version()
    latest_available = get_latest_dataset_batch()
    
    print(f"   Last Trained:  {last_trained}")
    print(f"   Latest Batch:  {latest_available}")
    
    if latest_available > last_trained:
        print("🚀 New batch detected! Triggering training...")
        # Pass the specific version to the job if needed, or just let it pick up latest
        # Ideally we pass it as an env var to the pod
        os.environ['TARGET_BATCH'] = latest_available
        submit_job()
    else:
        print("✅ already up to date.")

if __name__ == "__main__":
    main()
