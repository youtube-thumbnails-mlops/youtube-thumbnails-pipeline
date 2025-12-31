import wandb
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

PROJECT_NAME = "youtube-thumbnails-pipeline"
ENTITY = "daniele-acquaviva"
ARTIFACT_NAME = "thumbnail-classifier"
PROD_ALIAS = "prod"
LATEST_ALIAS = "latest"
TEST_SET_DIR = "/app/data/test_set" # Expecting DVC to pull this here
# YouTube Category ID Mapping (Must match train.py)
CATEGORY_MAP = {
    1: 0, 2: 1, 10: 2, 15: 3, 17: 4, 19: 5, 20: 6, 22: 7, 23: 8, 24: 9,
    25: 10, 26: 11, 27: 12, 28: 13, 29: 14, 30: 15, 43: 16
}
NUM_CLASSES = len(CATEGORY_MAP) # 17

import pandas as pd
from PIL import Image

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Load CSV
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Metadata file not found: {csv_file}")
            
        self.data = pd.read_csv(csv_file)
        
        # Filter: Ensure images exist
        self.data['img_path'] = self.data['video_id'].apply(
            lambda x: os.path.join(self.root_dir, f"{x}.jpg")
        )
        self.valid_data = [] # List of (path, label)
        
        missing_count = 0
        for _, row in self.data.iterrows():
            path = row['img_path']
            if os.path.exists(path):
                cat_id = int(row.get('category_id', 0))
                if cat_id in CATEGORY_MAP:
                    label = CATEGORY_MAP[cat_id]
                    self.valid_data.append((path, label))
            else:
                missing_count += 1
                
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} images from CSV not found in {root_dir}")
    
    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        path, label = self.valid_data[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            # Fallback (Gray) if corrupt
            image = Image.new('RGB', (224, 224), (128, 128, 128))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def promote_model():
    print(f"🕵️ Starting Active Evaluation for {ENTITY}/{PROJECT_NAME}...")
    
    if not os.path.exists(TEST_SET_DIR):
        print(f"⚠️ Golden Test Set not found at {TEST_SET_DIR}. Skipping Active Eval.")
        print("   Please create and pull 'test_set.dvc'.")
        return

    api = wandb.Api()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # 0. Prepare Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load CSV Dataset
    try:
        csv_path = os.path.join(TEST_SET_DIR, "metadata.csv")
        test_dataset = CSVDataset(root_dir=TEST_SET_DIR, csv_file=csv_path, transform=transform)
        
        if len(test_dataset) == 0:
            print("❌ Test set is empty (No valid image/CSV pairs).")
            return
            
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(f"   📉 Found {len(test_dataset)} test images (via CSV).")
    except Exception as e:
        print(f"❌ Failed to load test set: {e}")
        return

    # 1. Evaluate LATEST (Local file 'best_model.pth' from train.py)
    latest_path = "best_model.pth" 
    if not os.path.exists(latest_path):
        print("❌ 'best_model.pth' not found. Did training fail?")
        return
    
    print("   🧪 Evaluating LATEST model...")
    latest_model = load_result_model(latest_path, device)
    latest_acc = evaluate(latest_model, test_loader, device)
    print(f"   🆕 Latest Acc (Golden Set): {latest_acc:.4f}")

    # 2. Evaluate PROD (Fetch from W&B)
    print("   📥 Fetching PROD model for comparison...")
    prod_path = "prod_model.pth"
    try:
        prod_artifact = api.artifact(f"{ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME}:{PROD_ALIAS}")
        prod_dir = prod_artifact.download()
        # W&B usually downloads to artifact_dir/best_model.pth. We need to find it.
        # Assuming it's named 'best_model.pth' inside the artifact
        found_model = os.path.join(prod_dir, "best_model.pth")
        if os.path.exists(found_model):
            os.rename(found_model, prod_path)
            
            print("   🧪 Evaluating PROD model...")
            prod_model = load_result_model(prod_path, device)
            prod_acc = evaluate(prod_model, test_loader, device)
            print(f"   👑 Prod Acc (Golden Set):   {prod_acc:.4f}")
        else:
            print("   ⚠️ Prod artifact found but 'best_model.pth' missing. Treating as 0.0")
            prod_acc = 0.0

    except Exception as e:
        print(f"   ℹ️  No valid Prod model found ({e}). Treating baseline as 0.0.")
        prod_acc = 0.0

    # 3. Compare & Promote
    # Fetch the LATEST artifact object just to attach alias
    latest_artifact = api.artifact(f"{ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME}:{LATEST_ALIAS}")

    if latest_acc > prod_acc:
        print(f"🚀 PROMOTION! {latest_acc:.4f} > {prod_acc:.4f}")
        latest_artifact.aliases.append(PROD_ALIAS)
        latest_artifact.save()
        print(f"✅ 'prod' alias assigned to version {latest_artifact.version}")
    else:
        print(f"📉 No Promotion. {latest_acc:.4f} <= {prod_acc:.4f}")

if __name__ == "__main__":
    promote_model()
