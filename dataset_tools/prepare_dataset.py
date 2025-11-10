import os
import shutil
import csv
import random
from pathlib import Path

# Configuration
source_dir = r"E:\kinetics400_5per\train"
output_dir = os.path.join(os.getcwd(), "selected_videos")
csv_file = os.path.join(os.getcwd(), "video_labels.csv")
videos_per_class = None  # None means select all available videos

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# List to store video information
video_data = []

print(f"Scanning directories in: {source_dir}")
print(f"Output directory: {output_dir}")
print("-" * 60)

# Get all subdirectories (class folders)
class_folders = [f for f in os.listdir(source_dir) 
                 if os.path.isdir(os.path.join(source_dir, f))]

print(f"Found {len(class_folders)} class folders")

# Process each class folder
for class_name in class_folders:
    class_path = os.path.join(source_dir, class_name)
    
    # Get all mp4 files in this class folder
    mp4_files = [f for f in os.listdir(class_path) 
                 if f.lower().endswith('.mp4')]
    
    # Select all videos if videos_per_class is None, otherwise select up to the limit
    if videos_per_class is None:
        selected_videos = mp4_files
    else:
        selected_videos = mp4_files[:min(videos_per_class, len(mp4_files))]
    
    print(f"Class '{class_name}': Found {len(mp4_files)} videos, selecting {len(selected_videos)}")
    
    # Copy selected videos and record information
    for video_file in selected_videos:
        source_path = os.path.join(class_path, video_file)
        
        # Create unique filename to avoid conflicts
        # Format: classname_originalfilename.mp4
        new_filename = f"{class_name}_{video_file}"
        dest_path = os.path.join(output_dir, new_filename)
        
        # Copy the video file
        shutil.copy2(source_path, dest_path)
        
        # Store video information
        video_data.append({
            'filename': new_filename,
            'original_filename': video_file,
            'label': class_name,
            'source_path': source_path
        })

print("-" * 60)
print(f"Total videos copied: {len(video_data)}")

# Shuffle the videos in the output directory
print("\nShuffling videos in output directory...")
all_videos = os.listdir(output_dir)
temp_dir = os.path.join(os.getcwd(), "temp_shuffle")
os.makedirs(temp_dir, exist_ok=True)

# Move all videos to temp directory
for video in all_videos:
    shutil.move(os.path.join(output_dir, video), 
                os.path.join(temp_dir, video))

# Shuffle the list
random.shuffle(all_videos)

# Move videos back in shuffled order
for video in all_videos:
    shutil.move(os.path.join(temp_dir, video), 
                os.path.join(output_dir, video))

# Remove temp directory
os.rmdir(temp_dir)

# Shuffle the CSV data to match the shuffled videos
random.shuffle(video_data)

# Create CSV file with video labels
print(f"\nCreating CSV file: {csv_file}")
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'label', 'original_filename', 'source_path'])
    writer.writeheader()
    writer.writerows(video_data)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Videos copied: {len(video_data)}")
print(f"Output directory: {output_dir}")
print(f"CSV file: {csv_file}")
print(f"Classes processed: {len(class_folders)}")
print("\nDone!")
