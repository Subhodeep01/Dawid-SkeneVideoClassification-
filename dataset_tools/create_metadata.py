import csv
from collections import Counter

# Configuration
input_csv = "video_labels.csv"
output_metadata = "metadata_video_labels.txt"

print("=" * 60)
print("CREATING METADATA FOR video_labels.csv")
print("=" * 60)

# Read the CSV file
video_data = []
with open(input_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_data.append(row)

print(f"Total videos: {len(video_data)}")

# Count classes
class_counts = Counter(row['label'] for row in video_data)
total_classes = len(class_counts)

print(f"Total classes: {total_classes}")
print("-" * 60)

# Create metadata file
with open(output_metadata, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("VIDEO LABELS METADATA\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"Total Videos: {len(video_data)}\n")
    f.write(f"Total Classes: {total_classes}\n\n")
    
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
    
    f.write("\n" + "=" * 60 + "\n")
    f.write("CLASS DISTRIBUTION (By Video Count - Descending)\n")
    f.write("=" * 60 + "\n\n")
    
    for class_name, count in class_counts.most_common():
        f.write(f"{class_name}: {count} video(s)\n")

print(f"\nMetadata file created: {output_metadata}")
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total videos: {len(video_data)}")
print(f"Total classes: {total_classes}")
print(f"\nTop 10 classes by video count:")
for class_name, count in class_counts.most_common(10):
    print(f"  {class_name}: {count} videos")
print("\nDone!")
