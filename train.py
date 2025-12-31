import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# Config
PROJECT_NAME = "youtube-thumbnails-pipeline"
ENTITY = "daniele-acquaviva"

# YouTube Category ID Mapping (Simplified)
# Real mapping should be comprehensive or dynamic
CATEGORY_MAP = {
    1: 0, 2: 1, 10: 2, 15: 3, 17: 4, 19: 5, 20: 6, 22: 7, 23: 8, 24: 9,
    25: 10, 26: 11, 27: 12, 28: 13, 29: 14, 30: 15, 43: 16
}
NUM_CLASSES = len(CATEGORY_MAP)


class ThumbnailDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter missing images
        self.data['img_path'] = self.data['video_id'].apply(
            lambda x: os.path.join(self.img_dir, f"{x}.jpg")
        )
        # In a real scenario, we might want to verify file existence here
        # self.data = self.data[self.data['img_path'].apply(os.path.exists)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['img_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            image = Image.new('RGB', (224, 224), (128, 128, 128))
            
        # Get category ID and map to class index
        cat_id = int(row.get('category_id', 0))
        label = CATEGORY_MAP.get(cat_id, 0) # Default to class 0 if unknown
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def train(args):
    # 1. Initialize W&B
    run = wandb.init(
        project=PROJECT_NAME, 
        job_type="training",
        config=args
    )
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Setup Data
    print(f"Loading data from {args.data_dir}...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ThumbnailDataset(
        csv_file=os.path.join(args.data_dir, "metadata.csv"),
        img_dir=args.data_dir,
        transform=transform
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 3. Setup Model
    print("Setting up model (Classification)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace last layer for Classification
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)
    
    # FINE-TUNING LOGIC
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"🔄 Loading checkpoint from {args.checkpoint_path}...")
        try:
            state_dict = torch.load(args.checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Checkpoint loaded. Fine-tuning from previous state.")
        except Exception as e:
            print(f"⚠️ Checkpoint load failed (Architecture changed?): {e}")
            print("reshaping or ignoring mismatch is possible, but for safety:")
            print("🔄 Falling back to training from scratch (ImageNet weights).")
            # If we wanted to be fancy, we could partial load, but strict fallback is safer for automation
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 4. Train Loop
    print("Starting training...")
    best_loss = float('inf')
    best_acc = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        wandb.log({
            "train_loss": epoch_loss, "train_acc": epoch_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "epoch": epoch
        })
        
        # Checkpoint (Save if Accuracy Improves)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            
    # 5. Log Artifacts
    print("Training complete. Saving model...")
    
    # Save metadata for promotion script
    artifact = wandb.Artifact('thumbnail-classifier', type='model', metadata={"val_acc": best_acc})
    artifact.add_file('best_model.pth')
    wandb.log_artifact(artifact)
    
    # Export ONNX for Frontend (ONNX Runtime Web)
    print("Exporting ONNX model...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    # Opset 12 is a good balance for web compatibility
    torch.onnx.export(
        model, 
        dummy_input, 
        "model.onnx", 
        verbose=False,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    wandb.save("model.onnx")
    
    wandb.finish()

    print("✅ Training script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset containing images and metadata.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to initialize weights from")
    
    args = parser.parse_args()
    train(args)
