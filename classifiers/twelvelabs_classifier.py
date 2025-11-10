import os
import time
import csv
from pathlib import Path
from dotenv import load_dotenv
from twelvelabs import TwelveLabs
from typing import Any
from twelvelabs.indexes import IndexesCreateRequestModelsItem

# Load environment variables from .env file
load_dotenv()

# Configuration
SAMPLED_VIDEOS_DIR = "sampled_videos"
METADATA_FILE = "metadata.txt"
OUTPUT_CSV = "twelvelabs_predictions2.csv"
INDEX_NAME = "video_classification_index4"

# List of 60 classes from metadata.txt
CLASSES = [
    "abseiling",
    "applauding",
    "applying cream",
    "baby waking up",
    "balloon blowing",
    "bandaging",
    "bench pressing",
    "blasting sand",
    "canoeing or kayaking",
    "capoeira",
    "changing oil",
    "changing wheel",
    "cooking on campfire",
    "dancing ballet",
    "dancing charleston",
    "dancing macarena",
    "doing nails",
    "driving car",
    "dunking basketball",
    "feeding goats",
    "fixing hair",
    "frying vegetables",
    "hurdling",
    "javelin throw",
    "jogging",
    "juggling soccer ball",
    "laughing",
    "laying bricks",
    "lunge",
    "making snowman",
    "moving furniture",
    "plastering",
    "playing badminton",
    "playing chess",
    "playing didgeridoo",
    "playing keyboard",
    "playing trombone",
    "playing xylophone",
    "pole vault",
    "pumping fist",
    "pushing wheelchair",
    "riding elephant",
    "riding mountain bike",
    "riding unicycle",
    "ripping paper",
    "sharpening knives",
    "shuffling cards",
    "sign language interpreting",
    "skateboarding",
    "snatch weight lifting",
    "snorkeling",
    "spray painting",
    "squat",
    "swinging legs",
    "tango dancing",
    "trimming or shaving beard",
    "tying bow tie",
    "unloading truck",
    "vault",
    "waiting in line"
]

def setup_twelvelabs_api():
    """Setup Twelve Labs API with API key from environment variable"""
    api_key = os.getenv('TWELVELABS_API_KEY')
    
    if not api_key:
        print("ERROR: TWELVELABS_API_KEY not found in environment variables!")
        print("\nPlease add to your .env file:")
        print("TWELVELABS_API_KEY=your_api_key_here")
        print("\nGet your API key from: https://playground.twelvelabs.io/")
        return None
    
    client = TwelveLabs(api_key=api_key)
    return client

def get_or_create_index(client, name):
    """Get existing index or create a new one"""
    print(f"\nChecking for existing index: {name}")
    
    # List existing indexes
    indexes = list(client.indexes.list())
    
    for index in indexes:
        if index.index_name == name:
            print(f"Found existing index: {index.id}")
            print(f"Index models: {index.models}")
            
            # Check if index supports text generation
            model_names = [m.model_name for m in index.models]
            if 'pegasus1.2' in model_names or 'pegasus1' in model_names:
                print("Index supports text generation")
                return index.id
            else:
                print("Warning: Index does not support text generation (has marengo).")
                print("Creating new index with pegasus engine for text generation...")
                # Use a different name for the new index
                name = f"{name}_pegasus"
                break
    
    # Create new index if not found or doesn't support text generation
    print(f"Creating new index: {name}")
    index = client.indexes.create(
        index_name=name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2",
                model_options=["visual", "audio"]
            )
        ]
    )
    print(f"Created index: {index.id}")
    return index.id

def upload_video(client, index_id, video_path, video_filename):
    """Upload video to Twelve Labs"""
    print(f"  Uploading video to Twelve Labs...")
    
    # Open and upload the video file
    with open(video_path, 'rb') as video_file:
        task = client.tasks.create(
            index_id=index_id,
            video_file=video_file
        )
    
    # Wait for upload to complete
    print(f"    Task ID: {task.id}")
    
    # Poll for task completion
    while True:
        task_status = client.tasks.retrieve(task.id)
        print(f"    Status: {task_status.status}")
        
        if task_status.status == "ready":
            print(f"  Video uploaded successfully. Video ID: {task_status.video_id}")
            return task_status.video_id
        elif task_status.status == "failed":
            raise Exception(f"Upload failed: {task_status.status}")
        
        time.sleep(5)


def classify_video(client, index_id, video_id, classes):
    """Classify video using Twelve Labs Analyze API"""
    print(f"  Classifying video...")
    
    # Create classification prompt
    class_list = "\n".join([f"{i+1}. {cls}" for i, cls in enumerate(classes)])
    
    prompt = f"""Analyze this video and classify the main action being performed into ONE of the following {len(classes)} categories:

{class_list}

Instructions:
- Watch the video carefully and identify the primary action
- Choose ONLY ONE category that best matches the action
- Respond with ONLY the category name exactly as written above
- Do not include explanations or additional text

Classification:"""

    # Use Analyze API for classification
    res = client.analyze(
        video_id=video_id,
        prompt=prompt
    )
    
    # Extract the text from the response
    predicted_class = res.data.strip()
    
    # Validate prediction
    if predicted_class not in classes:
        print(f"    Warning: '{predicted_class}' not in class list. Finding best match...")
        # Try to find closest match (case-insensitive)
        predicted_class_lower = predicted_class.lower()
        for cls in classes:
            if cls.lower() == predicted_class_lower or cls.lower() in predicted_class_lower:
                predicted_class = cls
                break
    
    return predicted_class

def classify_all_videos(client, video_dir, classes, output_csv, start_from=1, end_at=None):
    """Classify all videos in the directory"""
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    if end_at:
        video_files = video_files[start_from-1:end_at]
    else:
        video_files = video_files[start_from-1:]
    
    total_videos = len(video_files)
    
    print("=" * 60)
    print(f"VIDEO CLASSIFICATION WITH TWELVE LABS API")
    print("=" * 60)
    print(f"Total videos to classify: {total_videos}")
    print(f"Number of classes: {len(classes)}")
    print(f"Results will be saved incrementally to: {output_csv}")
    print("-" * 60)
    
    # Get or create index
    index = get_or_create_index(client, INDEX_NAME)
    index_id = index.id
    
    results = []
    
    # Helper function to save results to CSV
    def save_results():
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'filename', 'predicted_class', 'twelvelabs_video_id'])
            writer.writeheader()
            writer.writerows(results)
    
    try:
        for idx, video_filename in enumerate(video_files, start=start_from):
            video_path = os.path.join(video_dir, video_filename)
            
            print(f"\n[{idx}/{start_from + total_videos - 1}] Processing: {video_filename}")
            
            try:
                # Upload video to Twelve Labs
                tl_video_id = upload_video(client, index_id, video_path, video_filename)
                
                # Classify the video
                predicted_class = classify_video(client, index_id, tl_video_id, classes)
                
                print(f"  Predicted class: {predicted_class}")
                
                # Store result
                results.append({
                    'video_id': int(video_filename.split('.')[0]),
                    'filename': video_filename,
                    'predicted_class': predicted_class,
                    'twelvelabs_video_id': tl_video_id
                })
                
                # Save results incrementally after each video
                save_results()
                print(f"  Progress saved to CSV ({len(results)} videos classified)")
                
                # Add a small delay to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                
                # Check for quota/token errors
                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print("\n" + "!" * 60)
                    print("API QUOTA/RATE LIMIT REACHED!")
                    print("!" * 60)
                    results.append({
                        'video_id': int(video_filename.split('.')[0]),
                        'filename': video_filename,
                        'predicted_class': 'QUOTA_EXHAUSTED',
                        'twelvelabs_video_id': 'N/A'
                    })
                    save_results()
                    print(f"Saved {len(results)} classifications before quota limit.")
                    print(f"You can resume later by starting from video {idx + 1}.")
                    raise
                
                results.append({
                    'video_id': int(video_filename.split('.')[0]),
                    'filename': video_filename,
                    'predicted_class': 'ERROR',
                    'twelvelabs_video_id': 'N/A'
                })
                
                # Save even on error
                save_results()
                print(f"  Progress saved to CSV ({len(results)} videos processed)")
    
    except KeyboardInterrupt:
        print("\n\n" + "!" * 60)
        print("INTERRUPTED BY USER!")
        print("!" * 60)
        save_results()
        print(f"Saved {len(results)} classifications before interruption.")
        print(f"You can resume later by starting from video {len(results) + start_from}.")
        return results
    
    except Exception as e:
        print("\n\n" + "!" * 60)
        print("UNEXPECTED ERROR!")
        print("!" * 60)
        print(f"Error: {str(e)}")
        save_results()
        print(f"Saved {len(results)} classifications before error.")
        raise
    
    # Final save
    print("\n" + "=" * 60)
    print("Saving final results to CSV...")
    save_results()
    
    print(f"Results saved to: {output_csv}")
    print("=" * 60)
    print("DONE!")
    
    return results

def main():
    # Setup Twelve Labs API
    client = setup_twelvelabs_api()
    if client is None:
        return
    
    print(f"\nTwelve Labs API initialized")
    print(f"Videos directory: {SAMPLED_VIDEOS_DIR}")
    print(f"Output CSV: {OUTPUT_CSV}")
    
    # Ask user for range
    print("\n" + "=" * 60)
    start = input("Enter starting video number (default 1): ").strip()
    start = int(start) if start else 1
    
    end = input("Enter ending video number (default: all remaining): ").strip()
    end = int(end) if end else None
    
    # Classify videos
    results = classify_all_videos(
        client=client,
        video_dir=SAMPLED_VIDEOS_DIR,
        classes=CLASSES,
        output_csv=OUTPUT_CSV,
        start_from=start,
        end_at=end
    )


# Wrapper function for LangGraph integration
def classify_video_twelvelabs(video_path, classes):
    """
    Wrapper function for LangGraph orchestrator.
    
    Args:
        video_path: Full path to the video file
        classes: List of class labels
    
    Returns:
        Predicted class name or "ERROR"
    """
    try:
        # Setup client
        client = setup_twelvelabs_api()
        if client is None:
            return "ERROR"
        
        # Get or create index
        index_id = get_or_create_index(client, INDEX_NAME)
        if index_id is None:
            return "ERROR"
        
        # Upload the video
        print(f"    Uploading video to Twelve Labs...")
        video_filename = os.path.basename(video_path)
        video_id = upload_video(client, index_id, video_path, video_filename)
        if video_id is None:
            return "ERROR"
        
        # Classify the video
        prediction = classify_video(client, index_id, video_id, classes)
        return prediction if prediction else "ERROR"
    
    except Exception as e:
        print(f"Error in classify_video_twelvelabs: {e}")
        return "ERROR"


if __name__ == "__main__":
    main()
