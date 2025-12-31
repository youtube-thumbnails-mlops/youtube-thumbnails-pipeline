#!/bin/bash
set -e

echo "🚀 RunPod Entrypoint Starting..."

# 1. Setup Environment
# Ensure we have the dataset
# If running in RunPod, we might need to clone the dataset repo first
# The container image expects code in /app, but data needs to be pulled.

DATA_ROOT="/workspace/data"
mkdir -p "$DATA_ROOT"

# Check if we are running locally or in cloud
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ℹ️  Local run detected (no RUNPOD_API_KEY). Assuming data is mounted or present."
    # Fallback to default
    exec "$@"
    exit 0
fi

echo "☁️  Cloud run detected. Setting up DVC..."

# 2. Clone Dataset Repo (If not mounted)
# We assume the container has the CODE (train repo), but needs the DATA (dataset repo).
# We use a public clone or token if private.
# Assuming public for now or passed via env if private.

REPO_DIR="/workspace/dataset_repo"
if [ ! -d "$REPO_DIR" ]; then
    echo "⬇️  Cloning dataset repository..."
    # Ideally use a token if private: https://$TOKEN@github.com/...
    # For now assuming public or simple clone
    git clone https://github.com/daniele-acquaviva/youtube-thumbnails-dataset.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# 3. Checkout Target Version
if [ -n "$TARGET_BATCH" ]; then
    echo "🎯 Checking out target batch: $TARGET_BATCH"
    # TARGET_BATCH is like "batch_005". The tag is "v5.0"
    # Or we can just checkout main if we want latest.
    # The user asked: "will it take the latest tag?" -> Check_and_trigger passes logic.
    
    # If check_and_trigger passes "batch_005", we find the tag.
    # Simple heuristic: v{batch_number}.0
    BATCH_NUM=$(echo "$TARGET_BATCH" | grep -oE '[0-9]+')
    TAG_NAME="v${BATCH_NUM}.0"
    
    echo "   (Mapping $TARGET_BATCH -> Tag $TAG_NAME)"
    git checkout "$TAG_NAME" || echo "⚠️ Tag $TAG_NAME not found, staying on main."
else
    echo "ℹ️  No TARGET_BATCH set, using latest (main)."
fi

# 4. Configure DVC (Credentials passed via Env)
dvc remote modify origin --local endpointurl "$R2_ENDPOINT"
dvc remote modify origin --local access_key_id "$AWS_ACCESS_KEY_ID"
dvc remote modify origin --local secret_access_key "$AWS_SECRET_ACCESS_KEY"

echo "⬇️  Pulling data (this may take a moment)..."
dvc pull

# 5. MERGE DATASETS
# We have batches/batch_*/ and current/
# train.py expects ONE folder. We merge into $DATA_ROOT/merged

MERGED_DIR="$DATA_ROOT/merged"
mkdir -p "$MERGED_DIR"
MERGED_CSV="$MERGED_DIR/metadata.csv"

echo "🔄 Merging daily batches into single training set..."

# Initialize CSV with header (take from first batch found)
# We now use the python script for robust merging
echo "🔄 Merging daily batches into single training set (using Python)..."

# Ensure we have the script (it's in /app/scripts usually, or we are in /app currently?)
# entrypoint starts in /app? No, "cd $REPO_DIR" happens earlier.
# The `scripts/merge_datasets.py` is in the CODE repository (/app), not the DATA repository ($REPO_DIR).
# So we run it from /app/scripts/merge_datasets.py

python /app/scripts/merge_datasets.py --source_dir . --output_file "$MERGED_CSV"

# Link images (Assume images are in same dir as csv)
# We still need to link images. The python script only handles CSV.
find batches -name "metadata.csv" | sort | while read csv; do
    DIR=$(dirname "$csv")
    cp -n "$DIR"/*.jpg "$MERGED_DIR/" 2>/dev/null || true
done

COUNT=$(($(wc -l < "$MERGED_CSV") - 1))
echo "✅ Merge complete. Total training samples: $COUNT"

# Fetch Model (Fine-Tuning)
echo "📥 Checking for previous model checkpoint..."
python /app/scripts/fetch_model.py

CHECKPOINT_ARG=""
if [ -f "checkpoint.pth" ]; then
    echo "✅ Found checkpoint, enabling fine-tuning."
    CHECKPOINT_ARG="--checkpoint_path checkpoint.pth"
fi

# 6. Run Training
# Pass the merged dir to train.py
echo "🚀 Starting Training..."
cd /app # Back to code
# Replace --data_dir arg or just run the command passed
# We override the command to force our data dir
exec python train.py --data_dir "$MERGED_DIR" --epochs 10 $CHECKPOINT_ARG
