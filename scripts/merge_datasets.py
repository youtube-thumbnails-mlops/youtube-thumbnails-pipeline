import os
import argparse
import pandas as pd
import glob
from pathlib import Path

def merge_datasets(source_dir, output_file):
    """
    Merges metadata.csv files from all batch directories in source_dir.
    Handles schema evolution by using pandas.concat (aligns columns, fills NaNs).
    """
    source_path = Path(source_dir)
    print(f"🔍 Searching for batches in {source_path}...")

    # Find all batches/batch_*/metadata.csv
    # We accept batches/batch_XXX/metadata.csv
    csv_files = list(source_path.glob("batches/batch_*/metadata.csv"))
    
    # Also optionally check if there are csvs in 'current' if the user wanted (though logic says frozen batches only)
    # The original script prioritized batches logic. We adhere to "batches/*".

    if not csv_files:
        print("⚠️  No batch CSV files found.")
        # Create empty CSV with basic headers to prevent crash if training checks
        # Assuming minimal columns needed
        pd.DataFrame(columns=['video_id', 'category_id', 'img_path']).to_csv(output_file, index=False)
        return

    print(f"found {len(csv_files)} batch files.")

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Add a 'batch_source' column for debugging/tracking if needed, or just merge
            # df['batch_source'] = f.parent.name 
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to read {f}: {e}")

    if not dfs:
        print("❌ No valid dataframes loaded.")
        return

    # ROBUST MERGE
    # pandas.concat connects them. sort=False prevents sorting columns alphabetically.
    # Columns not present in one DF but present in others will be filled with NaN.
    print("🔄 Concatenating dataframes...")
    merged_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    print(f"✅ Merged {len(merged_df)} rows. Columns: {list(merged_df.columns)}")
    
    # Save to output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"💾 Saved merged dataset to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Root directory containing batches/ folder")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the merged csv")
    
    args = parser.parse_args()
    merge_datasets(args.source_dir, args.output_file)
