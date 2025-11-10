"""
Script to upscale videos with resolution less than 360x360 to 360x360.
Videos already at or above 360x360 are left unchanged.
"""

import os
import cv2
from pathlib import Path
import subprocess
import shutil

def get_video_resolution(video_path):
    """Get the resolution of a video file."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def upscale_video(input_path, output_path, target_size=360):
    """Upscale video to target size using ffmpeg while maintaining aspect ratio."""
    # Get original dimensions
    cap = cv2.VideoCapture(input_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Upscaling {Path(input_path).name}...")
    print(f"  Original: {orig_width}x{orig_height}")
    
    # Calculate new dimensions maintaining aspect ratio
    if orig_width < orig_height:
        new_width = target_size
        new_height = int((orig_height / orig_width) * target_size)
    else:
        new_height = target_size
        new_width = int((orig_width / orig_height) * target_size)
    
    # Make dimensions divisible by 2 for video encoding
    new_width = new_width + (new_width % 2)
    new_height = new_height + (new_height % 2)
    
    print(f"  New: {new_width}x{new_height}")
    
    # Use ffmpeg to upscale
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale={new_width}:{new_height}',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'copy',
        '-y',  # Overwrite output file
        output_path
    ]
    
    # Run ffmpeg with minimal output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return False
    
    print(f"  ✓ Upscaled successfully")
    return True

def main():
    sampled_videos_dir = Path("sampled_videos")
    temp_dir = Path("temp_upscaled")
    
    if not sampled_videos_dir.exists():
        print(f"Error: {sampled_videos_dir} directory not found!")
        return
    
    # Create temporary directory for upscaled videos
    temp_dir.mkdir(exist_ok=True)
    
    # Get all video files
    video_files = list(sampled_videos_dir.glob("*.mp4"))
    
    print(f"Found {len(video_files)} videos to check...")
    print()
    
    upscaled_count = 0
    skipped_count = 0
    
    for video_file in video_files:
        try:
            # Get video resolution
            width, height = get_video_resolution(str(video_file))
            
            # Check if upscaling is needed
            if width < 360 or height < 360:
                print(f"{video_file.name}: {width}x{height} - UPSCALING")
                
                # Upscale to temp directory
                temp_output = temp_dir / video_file.name
                upscale_video(str(video_file), str(temp_output), target_size=360)
                
                # Replace original with upscaled version
                shutil.move(str(temp_output), str(video_file))
                
                upscaled_count += 1
                print()
            else:
                print(f"{video_file.name}: {width}x{height} - OK (skipping)")
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            continue
    
    # Clean up temp directory
    if temp_dir.exists():
        temp_dir.rmdir()
    
    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Upscaled: {upscaled_count} videos")
    print(f"  Skipped: {skipped_count} videos (already >= 360x360)")
    print("=" * 60)

if __name__ == "__main__":
    main()
