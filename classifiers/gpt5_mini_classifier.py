"""
Video Classification using GPT-5-mini Vision API
Processes videos from sampled_videos folder and classifies them into 60 action categories.
"""

import os
import csv
import base64
import cv2
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# The 60 classes from metadata.txt
CLASSES = [
    "abseiling", "applauding", "applying cream", "baby waking up", "balloon blowing",
    "bandaging", "bench pressing", "blasting sand", "canoeing or kayaking", "capoeira",
    "changing oil", "changing wheel", "cooking on campfire", "dancing ballet", "dancing charleston",
    "dancing macarena", "doing nails", "driving car", "dunking basketball", "feeding goats",
    "fixing hair", "frying vegetables", "hurdling", "javelin throw", "jogging",
    "juggling soccer ball", "laughing", "laying bricks", "lunge", "making snowman",
    "moving furniture", "plastering", "playing badminton", "playing chess", "playing didgeridoo",
    "playing keyboard", "playing trombone", "playing xylophone", "pole vault", "pumping fist",
    "pushing wheelchair", "riding elephant", "riding mountain bike", "riding unicycle", "ripping paper",
    "sharpening knives", "shuffling cards", "sign language interpreting", "skateboarding", "snatch weight lifting",
    "snorkeling", "spray painting", "squat", "swinging legs", "tango dancing",
    "trimming or shaving beard", "tying bow tie", "unloading truck", "vault", "waiting in line"
]

def extract_frames(video_path, num_frames=8):
    """
    Extract evenly spaced frames from video for gpt-5-mini analysis.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 8 for good coverage)
    
    Returns:
        List of base64 encoded frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Calculate frame indices to extract
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    frames_b64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Convert to base64
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frames_b64.append(frame_b64)
    
    cap.release()
    return frames_b64

def classify_video_gpt5(video_path, classes, max_retries=3):
    """
    Classify video using gpt-5-mini Vision API.
    
    Args:
        video_path: Path to video file
        classes: List of possible class labels
        max_retries: Number of retry attempts
    
    Returns:
        Predicted class name or "ERROR"
    """
    print(f"  Classifying with gpt-5-mini Vision...")
    
    for attempt in range(max_retries):
        try:
            # Extract frames from video
            frames = extract_frames(video_path, num_frames=8)
            
            if not frames:
                print(f"  ERROR: Could not extract frames from video")
                return "ERROR"
            
            # Build the messages with multiple frames
            content = [
                {
                    "type": "text",
                    "text": f"""Analyze this video sequence (shown as {len(frames)} frames) and classify the main action being performed.

Choose ONLY ONE action from this list:
{', '.join(classes)}

Respond with ONLY the action name from the list above, nothing else. If you're unsure, choose the closest match."""
                }
            ]
            
            # Add all frames
            for frame_b64 in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "low"  # Use low detail to reduce cost
                    }
                })
            
            # Call GPT-5-mini API
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{
                    "role": "user",
                    "content": content
                }],
                
            )
            
            # Extract prediction
            prediction = response.choices[0].message.content.strip()
            
            # Clean up the prediction (remove quotes, periods, etc.)
            prediction = prediction.strip('."\'')
            
            # Validate prediction is in our classes
            if prediction in classes:
                print(f"  ✓ Predicted: {prediction}")
                return prediction
            else:
                # Try to find closest match
                prediction_lower = prediction.lower()
                for cls in classes:
                    if cls.lower() in prediction_lower or prediction_lower in cls.lower():
                        print(f"  ✓ Predicted: {cls} (matched from: {prediction})")
                        return cls
                
                # If no match found, return the raw prediction
                print(f"  ⚠ Predicted: {prediction} (not in class list)")
                return prediction
        
        except Exception as e:
            error_msg = str(e)
            print(f"  Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            # Check if it's a rate limit error
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                # Extract wait time from error message if available
                import re
                wait_match = re.search(r'try again in ([\d.]+)s', error_msg)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 1  # Add 1 second buffer
                else:
                    wait_time = 10  # Default wait for rate limits
                print(f"  Rate limit hit. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                continue
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ERROR: All retry attempts failed")
                import traceback
                traceback.print_exc()
                return "ERROR"
    
    return "ERROR"

def process_all_videos(video_dir='sampled_videos', output_csv='gpt-5-mini_predictions2.csv', start_from=1):
    """
    Process all videos in the directory and save predictions to CSV.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Output CSV file path
        start_from: Video number to start from (1-indexed, default: 1)
    """
    # Get all video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    total_videos = len(video_files)
    
    # Validate start_from parameter
    if start_from < 1:
        start_from = 1
    if start_from > total_videos:
        print(f"ERROR: start_from ({start_from}) is greater than total videos ({total_videos})")
        return
    
    # Determine if we're resuming or starting fresh
    mode = "append" if start_from > 1 else "write"
    file_mode = 'a' if mode == "append" else 'w'
    
    print(f"\n{'='*60}")
    print(f"gpt-5-mini Video Classification")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"Starting from video: {start_from}")
    print(f"Videos to process: {total_videos - start_from + 1}")
    print(f"Mode: {'Resuming' if mode == 'append' else 'Starting fresh'}")
    print(f"Output file: {output_csv}")
    print(f"{'='*60}\n")
    
    # Open CSV file for writing or appending
    with open(output_csv, file_mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if starting fresh
        if mode == "write":
            writer.writerow(['video_name', 'predicted_class'])
        
        # Process videos starting from the specified index
        start_time = time.time()
        videos_to_process = video_files[start_from - 1:]  # Convert to 0-indexed
        
        for idx, video_file in enumerate(videos_to_process, start_from):
            video_path = os.path.join(video_dir, video_file)
            
            print(f"[{idx}/{total_videos}] Processing: {video_file}")
            
            # Classify video
            predicted_class = classify_video_gpt5(video_path, CLASSES)
            
            # Write to CSV
            writer.writerow([video_file, predicted_class])
            csvfile.flush()  # Ensure data is written immediately
            
            # Small delay to avoid rate limits (0.5 seconds between videos)
            time.sleep(0.5)
            
            # Progress update
            elapsed = time.time() - start_time
            videos_processed = idx - start_from + 1
            avg_time = elapsed / videos_processed
            remaining = (total_videos - idx) * avg_time
            
            print(f"  Progress: {idx}/{total_videos} ({idx/total_videos*100:.1f}%)")
            print(f"  Estimated time remaining: {remaining/60:.1f} minutes\n")
    
    total_time = time.time() - start_time
    videos_processed = total_videos - start_from + 1
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Videos processed: {videos_processed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per video: {total_time/videos_processed:.1f} seconds")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not found in .env file")
        print("Please add your OpenAI API key to .env file:")
        print("OPENAI_API_KEY=your-key-here")
        exit(1)
    
    # Get start_from parameter from command line if provided
    start_from = 1
    if len(sys.argv) > 1:
        try:
            start_from = int(sys.argv[1])
            print(f"Starting from video number: {start_from}")
        except ValueError:
            print(f"Invalid start number '{sys.argv[1]}', using default (1)")
    
    # Process all videos
    process_all_videos(start_from=start_from)
