"""
Video Classification using Qwen-VL (Alibaba Cloud Model Studio)
Uses DashScope SDK for direct video input to Qwen-VL models.
Processes videos from sampled_videos folder and classifies them into 60 action categories.
"""

import os
import csv
import time
import dashscope
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alibaba Cloud DashScope API configuration
# dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
# Set base URL for region (default is Beijing)
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
# For Singapore region, use:
# dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

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

def upload_video_to_oss(video_path):
    """
    Upload video to a temporary URL or convert to base64.
    For DashScope API, we need to provide video as a URL.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Video URL or path
    """
    # For now, return the local file path
    # In production, you would upload to OSS (Object Storage Service)
    # and return the public URL
    return f"file://{os.path.abspath(video_path)}"

def classify_video_qwen(video_path, classes, max_retries=3):
    """
    Classify video using Qwen-VL via DashScope SDK.
    
    Args:
        video_path: Path to video file
        classes: List of possible class labels
        max_retries: Number of retry attempts
    
    Returns:
        Predicted class name or None
    """
    for attempt in range(max_retries):
        try:
            # Prepare the prompt
            prompt_text = f"""Analyze this video and identify the main action being performed.

Choose ONLY ONE action from this list:
{chr(10).join(classes)}

Respond with ONLY the action name from the list above, nothing else."""
            
            # Build messages using DashScope SDK format
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'video': f"file://{os.path.abspath(video_path)}"},
                        {'text': prompt_text}
                    ]
                }
            ]
            
            # Call Qwen-VL using DashScope SDK
            response = dashscope.MultiModalConversation.call(
                api_key = os.getenv('DASHSCOPE_API_KEY'),
                model='qwen2.5-vl-3b-instruct',
                messages=messages,
                # max_tokens=50,
                # temperature=0.1
            )
            
            # Check response
            if response.status_code != 200:
                error_msg = response.message if hasattr(response, 'message') else str(response)
                print(f"    API Error ({response.status_code}): {error_msg[:300]}")
                
                # Check for rate limit
                if response.status_code == 429:
                    print(f"    Rate limit hit. Waiting 5 seconds...")
                    time.sleep(5)
                    continue
                
                return None
            
            # Extract prediction from response
            if hasattr(response, 'output') and 'choices' in response.output:
                choices = response.output['choices']
                if len(choices) > 0 and 'message' in choices[0]:
                    content = choices[0]['message']['content']
                    
                    # Handle content being a list or dict
                    if isinstance(content, list):
                        # Extract text from list of content items
                        prediction = ''
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                prediction += item['text']
                            elif isinstance(item, str):
                                prediction += item
                        prediction = prediction.strip()
                    elif isinstance(content, dict) and 'text' in content:
                        prediction = content['text'].strip()
                    else:
                        prediction = str(content).strip()
                    
                    # Clean up the prediction
                    prediction = prediction.strip('."\'').strip().lower()
                    
                    # Validate prediction is in our classes (case-insensitive matching)
                    for cls in classes:
                        if cls.lower() == prediction or cls.lower() in prediction or prediction in cls.lower():
                            return cls
                    
                    # Return raw prediction if no match
                    return prediction
            
            print(f"    Unexpected response format: {response}")
            return None
        
        except Exception as e:
            error_msg = str(e)
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                return None
    
    return None

def process_video(video_path, classes):
    """
    Process a single video and classify it using Qwen-VL.
    
    Args:
        video_path: Path to video file
        classes: List of possible class labels
    
    Returns:
        Predicted class name or "ERROR"
    """
    print(f"  Classifying with Qwen-VL (DashScope SDK)...")
    
    try:
        # Classify video directly
        prediction = classify_video_qwen(video_path, classes)
        
        if prediction:
            print(f"  ✓ Predicted: {prediction}")
            return prediction
        else:
            print(f"  ERROR: Classification failed")
            return "ERROR"
    
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return "ERROR"

def process_all_videos(video_dir='sampled_videos', output_csv='qwen_predictions2.csv', start_from=1):
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
    print(f"Qwen-VL Video Classification (DashScope SDK)")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"Starting from video: {start_from}")
    print(f"Videos to process: {total_videos - start_from + 1}")
    print(f"Mode: {'Resuming' if mode == 'append' else 'Starting fresh'}")
    print(f"Model: qwen2.5-vl-3b-instruct (native video support via SDK)")
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
            predicted_class = process_video(video_path, CLASSES)
            
            # Write to CSV
            writer.writerow([video_file, predicted_class])
            csvfile.flush()  # Ensure data is written immediately
            
            # Small delay between videos
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
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not found in .env file")
        print("Please add your Alibaba Cloud API key to .env file:")
        print("DASHSCOPE_API_KEY=your-key-here")
        print("\nGet your key from: https://dashscope.console.aliyun.com/apiKey")
        exit(1)
    
    # Verify API key format
    print(f"API Key loaded: {api_key[:10]}...{api_key[-4:]}")
    print(f"API Key length: {len(api_key)} characters")
    
    if not api_key.startswith('sk-'):
        print("\nWARNING: API key should start with 'sk-'")
        print("Please verify your API key from: https://dashscope.console.aliyun.com/apiKey")
        print("\nNote: You may need to:")
        print("1. Create a new API key")
        print("2. Ensure you're using the correct region (Beijing vs Singapore)")
        print("3. Verify the model (qwen-vl-max) is available for your account")
        exit(1)
    
    # Get start_from parameter from command line if provided
    start_from = 349
    if len(sys.argv) > 1:
        try:
            start_from = int(sys.argv[1])
            print(f"Starting from video number: {start_from}")
        except ValueError:
            print(f"Invalid start number '{sys.argv[1]}', using default (1)")
    
    # Process all videos
    process_all_videos(start_from=start_from)
