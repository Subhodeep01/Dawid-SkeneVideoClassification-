import os
import shutil
import csv
import random
import pandas as pd
from pathlib import Path

# Configuration
source_csv = "video_labels.csv"
source_dir = "selected_videos"
output_dir = "sampled_videos"
labels_csv = "sampled_labels.csv"
metadata_file = "metadata.txt"

max_classes = 60
total_samples = 1000

# List of classes to exclude from sampling (add more as needed)
excluded_classes = [
    "waxing chest",
    "grinding meat",
    "massaging back",
    "belly dancing",
    "milking cow"
    "waxing back",
    # "waxing eyebrows",
    # Add more classes here...
]

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("RANDOM VIDEO SAMPLING")
print("=" * 60)
print(f"Source: {source_csv}")
print(f"Target samples: {total_samples}")
print(f"Max classes: {max_classes}")
print("-" * 60)

# Read the original CSV
df = pd.read_csv(source_csv)

# Exclude specified classes
df = df[~df['label'].isin(excluded_classes)]
print(f"Excluded classes: {', '.join(excluded_classes)}")

# Get all unique classes
all_classes = df['label'].unique()
print(f"Total classes available (after exclusions): {len(all_classes)}")

# Randomly select up to 40 classes
selected_classes = random.sample(list(all_classes), min(max_classes, len(all_classes)))
print(f"Randomly selected classes: {len(selected_classes)}")

# Filter dataframe to only include selected classes
df_filtered = df[df['label'].isin(selected_classes)]
print(f"Videos in selected classes: {len(df_filtered)}")

# Randomly sample 200 videos from these classes
if len(df_filtered) < total_samples:
    print(f"Warning: Only {len(df_filtered)} videos available. Sampling all of them.")
    sampled_df = df_filtered
else:
    sampled_df = df_filtered.sample(n=total_samples, random_state=42)

print(f"Videos sampled: {len(sampled_df)}")
print("-" * 60)

# Create anonymized copies and tracking data
sampled_data = []
class_counts = {}

for idx, (_, row) in enumerate(sampled_df.iterrows(), start=1):
    # Create anonymized filename (e.g., 001.mp4, 002.mp4, etc.)
    new_filename = f"{idx:03d}.mp4"
    
    source_path = os.path.join(source_dir, row['filename'])
    dest_path = os.path.join(output_dir, new_filename)
    
    # Copy the video file
    if os.path.exists(source_path):
        shutil.copy2(source_path, dest_path)
        
        # Track data
        sampled_data.append({
            'video_id': idx,
            'filename': new_filename,
            'label': row['label'],
            'original_filename': row['filename']
        })
        
        # Count classes
        label = row['label']
        class_counts[label] = class_counts.get(label, 0) + 1
        
        if idx % 50 == 0:
            print(f"Processed {idx} videos...")
    else:
        print(f"Warning: Source file not found: {source_path}")

print("-" * 60)
print(f"Total videos copied: {len(sampled_data)}")

# Create labels CSV (video_id, filename, label)
print(f"\nCreating labels CSV: {labels_csv}")
with open(labels_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['video_id', 'filename', 'label', 'original_filename'])
    writer.writeheader()
    writer.writerows(sampled_data)

# Create metadata file
print(f"Creating metadata file: {metadata_file}")
with open(metadata_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("VIDEO DATASET METADATA\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Total Videos: {len(sampled_data)}\n")
    f.write(f"Total Classes: {len(class_counts)}\n\n")
    
    f.write("=" * 60 + "\n")
    f.write("CLASS DISTRIBUTION\n")
    f.write("=" * 60 + "\n\n")
    
    # Sort classes alphabetically
    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        f.write(f"{class_name}: {count} video(s)\n")
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("ALL CLASS NAMES (Alphabetically)\n")
    f.write("=" * 60 + "\n\n")
    
    for idx, class_name in enumerate(sorted(class_counts.keys()), start=1):
        f.write(f"{idx}. {class_name}\n")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Sampled videos: {len(sampled_data)}")
print(f"Output directory: {output_dir}")
print(f"Labels CSV: {labels_csv}")
print(f"Metadata file: {metadata_file}")
print(f"Total classes: {len(class_counts)}")
print("\nClass distribution:")
for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {class_name}: {count} videos")
if len(class_counts) > 10:
    print(f"  ... and {len(class_counts) - 10} more classes")
print("\nDone!")
