import wandb
import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os

# Config
ENTITY = "daniele5"
PROJECT = "model-registry"
MODEL_NAME = "thumbnail-classifier"
# Categories (ID -> Name mapping would be better, but using indices for now)
CATEGORY_MAP = {
    0: "Automobiles", 1: "Comedy", 2: "Education", 3: "Entertainment", 
    4: "Film & Animation", 5: "Gaming", 6: "Howto & Style", 7: "Music", 
    8: "News & Politics", 9: "Nonprofits & Activism", 10: "People & Blogs", 
    11: "Pets & Animals", 12: "Science & Technology", 13: "Shows", 
    14: "Sports", 15: "Travel & Events", 16: "Movies"
}

def verify_inference():
    print(f"🚀 Starting End-to-End Inference Test...")
    
    # 1. Download Model from Registry
    api = wandb.Api()
    print(f"📥 Fetching '{MODEL_NAME}:production' from W&B...")
    try:
        artifact = api.artifact(f"{ENTITY}/{PROJECT}/{MODEL_NAME}:production")
        artifact_dir = artifact.download()
        model_path = os.path.join(artifact_dir, "model.onnx")
        print(f"✅ Model downloaded to: {model_path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return

    # 2. Download Test Image (A Gaming Thumbnail example)
    img_url = "https://images.unsplash.com/photo-1542751371-adc38448a05e?auto=format&fit=crop&w=800&q=80" # Gaming/Esports setup
    print(f"🖼️  Downloading Test Image from: {img_url}")
    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        print(f"✅ Image Loaded: {img.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    # 3. Preprocess (Standard ImageNet: Resize 224, Normalize)
    print("⚙️  Preprocessing...")
    img = img.resize((224, 224))
    img_data = np.array(img).astype('float32') / 255.0
    img_data = img_data.transpose(2, 0, 1) # HWC -> CHW
    
    # Normalize (Mean/Std for ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    
    # Add batch dim
    input_tensor = img_data[np.newaxis, :]
    print(f"   Input Shape: {input_tensor.shape}")

    # 4. Run Inference
    print("🧠 Running ONNX Inference...")
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        logits = session.run([output_name], {input_name: input_tensor})[0]
        
        # Softmax
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
            
        probs = softmax(logits[0])
        
        # Top 3
        top3_indices = probs.argsort()[-3:][::-1]
        
        print("\n🏆 Predictions:")
        print("(Note: Model head is currently UNTRAINED, so predictions are random)")
        print("-" * 30)
        for i in top3_indices:
            cls_name = CATEGORY_MAP.get(i, f"Class {i}")
            print(f"   {i}: {cls_name:<20} {probs[i]*100:.2f}%")
        print("-" * 30)
        print("✅ Inference Pipeline Works!")
        
    except Exception as e:
        print(f"❌ Inference Failed: {e}")

if __name__ == "__main__":
    verify_inference()
