"""
Video Classification using Replicate API with LLaVA-13b
Extracts frames from videos and uses LLaVA-13b (image model) for classification.
Processes videos from sampled_videos folder and classifies them into 60 action categories.
"""

import os
import csv
import time
import cv2
import replicate
from pathlib import Path
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Initialize Replicate client
replicate_client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

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

def extract_frames(video_path, num_frames=3):
    """
    Extract evenly spaced frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default 3 for cost efficiency)
    
    Returns:
        List of frame file paths
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    # Calculate frame indices to extract
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    # Create temp directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_paths = []
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Save frame as image
            frame_path = os.path.join(temp_dir, f"{video_name}_frame_{idx}.jpg")
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths

def classify_frame_llava(frame_path, classes, max_retries=3):
    """
    Classify a single frame using LLaVA-13b.
    
    Args:
        frame_path: Path to frame image
        classes: List of possible class labels
        max_retries: Number of retry attempts
    
    Returns:
        Predicted class name or None
    """
    for attempt in range(max_retries):
        try:
            prompt = f"""What action is being performed in this image?

Choose ONLY ONE action from this list:
{', '.join(classes)}

Respond with ONLY the action name from the list above, nothing else."""
            
            # Run LLaVA-13b prediction
            output = replicate_client.run(
                "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
                input={
                    "image": open(frame_path, "rb"),
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.2
                }
            )
            
            # Extract prediction from output (handle generator)
            if hasattr(output, '__iter__') and not isinstance(output, str):
                # It's a generator or list, join all parts
                prediction = ''.join(str(x) for x in output).strip()
            else:
                prediction = str(output).strip()
            
            # Clean up the prediction
            prediction = prediction.strip('."\'').strip().lower()
            
            # Validate prediction is in our classes (case-insensitive matching)
            for cls in classes:
                if cls.lower() == prediction or cls.lower() in prediction or prediction in cls.lower():
                    return cls
            
            # Return raw prediction if no match found
            return prediction
        
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                import re
                wait_match = re.search(r'try again in ([\d.]+)s', error_msg)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 1
                else:
                    wait_time = 5
                time.sleep(wait_time)
                continue
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return None
    
    return None

def classify_video_replicate(video_path, classes, num_frames=3):
    """
    Classify video by extracting frames and using LLaVA-13b.
    
    Args:
        video_path: Path to video file
        classes: List of possible class labels
        num_frames: Number of frames to extract and classify
    
    Returns:
        Predicted class name or "ERROR"
    """
    print(f"  Classifying with LLaVA-13b (frame extraction)...")
    
    try:
        # Extract frames
        frame_paths = extract_frames(video_path, num_frames)
        
        if not frame_paths:
            print(f"  ERROR: Could not extract frames from video")
            return "ERROR"
        
        print(f"  Extracted {len(frame_paths)} frames")
        
        # Classify each frame
        predictions = []
        for idx, frame_path in enumerate(frame_paths):
            print(f"    Classifying frame {idx+1}/{len(frame_paths)}...", end=" ")
            prediction = classify_frame_llava(frame_path, classes)
            
            if prediction:
                predictions.append(prediction)
                print(f"✓ {prediction}")
            else:
                print(f"✗ Failed")
            
            # Clean up frame file
            try:
                os.remove(frame_path)
            except:
                pass
            
            # Small delay between frames
            time.sleep(0.3)
        
        # Aggregate predictions (majority vote)
        if predictions:
            # Count occurrences
            vote_counts = Counter(predictions)
            # Get most common prediction
            final_prediction = vote_counts.most_common(1)[0][0]
            print(f"  ✓ Final prediction (majority vote): {final_prediction}")
            print(f"    Vote breakdown: {dict(vote_counts)}")
            return final_prediction
        else:
            print(f"  ERROR: No successful frame classifications")
            return "ERROR"
    
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return "ERROR"

def process_all_videos(video_dir='sampled_videos', output_csv='replicate_predictions.csv'):
    """
    Process all videos in the directory and save predictions to CSV.
    
    Args:
        video_dir: Directory containing video files
        output_csv: Output CSV file path
    """
    # Get all video files
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    total_videos = len(video_files)
    
    print(f"\n{'='*60}")
    print(f"LLaVA-13b Video Classification (Frame Extraction)")
    print(f"{'='*60}")
    print(f"Total videos to process: {total_videos}")
    print(f"Frames per video: 3")
    print(f"Output file: {output_csv}")
    # print(f"Estimated cost: ~$0.001 per video (~$1 for 1000 videos)")
    print(f"{'='*60}\n")
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['video_name', 'predicted_class'])
        
        # Process each video
        start_time = time.time()
        for idx, video_file in enumerate(video_files, 1):
            video_path = os.path.join(video_dir, video_file)
            
            print(f"[{idx}/{total_videos}] Processing: {video_file}")
            
            # Classify video
            predicted_class = classify_video_replicate(video_path, CLASSES, num_frames=3)
            
            # Write to CSV
            writer.writerow([video_file, predicted_class])
            csvfile.flush()  # Ensure data is written immediately
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (total_videos - idx) * avg_time
            
            print(f"  Progress: {idx}/{total_videos} ({idx/total_videos*100:.1f}%)")
            print(f"  Estimated time remaining: {remaining/60:.1f} minutes\n")
    
    # Clean up temp directory
    try:
        os.rmdir("temp_frames")
    except:
        pass
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per video: {total_time/total_videos:.1f} seconds")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv('REPLICATE_API_TOKEN'):
        print("ERROR: REPLICATE_API_TOKEN not found in .env file")
        print("Please add your Replicate API token to .env file:")
        print("REPLICATE_API_TOKEN=your-token-here")
        print("\nGet your token from: https://replicate.com/account/api-tokens")
        exit(1)
    
    # Process all videos
    process_all_videos()
