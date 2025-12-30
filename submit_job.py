
import os
import runpod
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("RUNPOD_API_KEY")
runpod.api_key = API_KEY

def submit_job():
    # Configuration
    GPU_TYPE = "NVIDIA GeForce RTX 3090"
    IMAGE_NAME = "danieleacquaviva/youtube-thumbnails-train:latest" # Example
    
    print("🚀 Submitting job to RunPod...")
    
    # 1. Create a Pod
    pod = runpod.create_pod(
        name="bg-train-thumbnails",
        image_name=IMAGE_NAME,
        gpu_type_id=GPU_TYPE,
        gpu_count=1,
        container_disk_in_gb=20,
        env={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "DVC_REMOTE_URL": os.getenv("R2_ENDPOINT"),
            "AWS_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY"),
            # DVC often uses standard AWS env vars or specific config
        },
        docker_args="python train.py --epochs 10" # Command to run
    )
    
    print(f"✅ Pod created: {pod['id']}")
    print("Monitor status in RunPod dashboard or W&B.")

if __name__ == "__main__":
    submit_job()
