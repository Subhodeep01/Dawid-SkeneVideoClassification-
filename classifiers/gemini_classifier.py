import os
import google.generativeai as genai
from pathlib import Path
import time
import csv
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
SAMPLED_VIDEOS_DIR = "sampled_videos"
METADATA_FILE = "metadata.txt"
OUTPUT_CSV = "gemini_predictions.csv"

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

def setup_gemini_api():
    """Setup Gemini API with API key from environment variable"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment variables!")
        print("\nPlease create a .env file in the current directory with:")
        print("GEMINI_API_KEY=your_api_key_here")
        print("\nOr set the environment variable manually.")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def upload_video_to_gemini(video_path):
    """Upload video file to Gemini"""
    print(f"Uploading video: {video_path}")
    video_file = genai.upload_file(path=video_path)
    
    # Wait for the file to be processed
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
    
    if video_file.state.name == "FAILED":
        raise ValueError(f"Video processing failed: {video_file.state.name}")
    
    print(" Done!")
    return video_file

def classify_video(model, video_file, classes):
    """Classify video using Gemini API"""
    # Create the prompt with class list
    class_list = "\n".join([f"{i+1}. {cls}" for i, cls in enumerate(classes)])
    
    prompt = f"""You are a video classification expert. Analyze this video and classify it into ONE of the following {len(classes)} action classes:

{class_list}

Instructions:
1. Watch the video carefully and identify the main action being performed
2. Choose ONLY ONE class from the list above that best matches the action in the video
3. Respond with ONLY the class name, nothing else
4. The class name must be exactly as written in the list above

Your classification:"""

    # Generate content with video and prompt
    response = model.generate_content([video_file, prompt])
    
    predicted_class = response.text.strip()
    
    # Validate that the prediction is one of the valid classes
    if predicted_class not in classes:
        print(f"  Warning: Predicted class '{predicted_class}' not in class list. Using best match...")
        # Try to find closest match (case-insensitive)
        predicted_class_lower = predicted_class.lower()
        for cls in classes:
            if cls.lower() == predicted_class_lower:
                predicted_class = cls
                break
    
    return predicted_class

def classify_all_videos(model, video_dir, classes, output_csv, start_from=1, end_at=None):
    """Classify all videos in the directory"""
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])
    
    if end_at:
        video_files = video_files[start_from-1:end_at]
    else:
        video_files = video_files[start_from-1:]
    
    total_videos = len(video_files)
    
    print("=" * 60)
    print(f"VIDEO CLASSIFICATION WITH GEMINI API")
    print("=" * 60)
    print(f"Total videos to classify: {total_videos}")
    print(f"Number of classes: {len(classes)}")
    print(f"Results will be saved incrementally to: {output_csv}")
    print("-" * 60)
    
    results = []
    
    # Helper function to save results to CSV
    def save_results():
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['video_id', 'filename', 'predicted_class'])
            writer.writeheader()
            writer.writerows(results)
    
    try:
        for idx, video_filename in enumerate(video_files, start=start_from):
            video_path = os.path.join(video_dir, video_filename)
            
            print(f"\n[{idx}/{start_from + total_videos - 1}] Processing: {video_filename}")
            
            try:
                # Upload video to Gemini
                video_file = upload_video_to_gemini(video_path)
                
                # Classify the video
                predicted_class = classify_video(model, video_file, classes)
                
                print(f"  Predicted class: {predicted_class}")
                
                # Store result
                results.append({
                    'video_id': int(video_filename.split('.')[0]),
                    'filename': video_filename,
                    'predicted_class': predicted_class
                })
                
                # Delete the uploaded file to free up space
                genai.delete_file(video_file.name)
                print(f"  Deleted uploaded file from Gemini")
                
                # Save results incrementally after each video
                save_results()
                print(f"  Progress saved to CSV ({len(results)} videos classified)")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                
                # Check for quota/token errors
                if "quota" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                    print("\n" + "!" * 60)
                    print("API QUOTA EXHAUSTED!")
                    print("!" * 60)
                    results.append({
                        'video_id': int(video_filename.split('.')[0]),
                        'filename': video_filename,
                        'predicted_class': 'QUOTA_EXHAUSTED'
                    })
                    save_results()
                    print(f"Saved {len(results)} classifications before quota limit.")
                    print("You can resume later by starting from video {}.".format(idx + 1))
                    raise  # Re-raise to exit the loop
                
                results.append({
                    'video_id': int(video_filename.split('.')[0]),
                    'filename': video_filename,
                    'predicted_class': 'ERROR'
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
        print("You can resume later by starting from video {}.".format(len(results) + start_from))
        return results
    
    except Exception as e:
        print("\n\n" + "!" * 60)
        print("UNEXPECTED ERROR!")
        print("!" * 60)
        print(f"Error: {str(e)}")
        save_results()
        print(f"Saved {len(results)} classifications before error.")
        raise
    
    # Final save (redundant but ensures everything is saved)
    print("\n" + "=" * 60)
    print("Saving final results to CSV...")
    save_results()
    
    print(f"Results saved to: {output_csv}")
    print("=" * 60)
    print("DONE!")
    
    return results

def main():
    # Setup Gemini API
    model = setup_gemini_api()
    if model is None:
        return
    
    print(f"\nGemini Model: {model.model_name}")
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
        model=model,
        video_dir=SAMPLED_VIDEOS_DIR,
        classes=CLASSES,
        output_csv=OUTPUT_CSV,
        start_from=start,
        end_at=end
    )


# Wrapper function for LangGraph integration
def classify_video_gemini(video_path, classes):
    """
    Wrapper function for LangGraph orchestrator.
    
    Args:
        video_path: Full path to the video file
        classes: List of class labels
    
    Returns:
        Predicted class name or "ERROR"
    """
    try:
        # Setup model (will use cached if available)
        model = setup_gemini_api()
        if model is None:
            return "ERROR"
        
        # Upload the video to Gemini first
        video_file = upload_video_to_gemini(video_path)
        
        # Classify the single video with the uploaded file
        prediction = classify_video(model, video_file, classes)
        
        # Delete the uploaded file to free up space
        try:
            genai.delete_file(video_file.name)
        except:
            pass  # Ignore deletion errors
        
        return prediction if prediction else "ERROR"
    
    except Exception as e:
        print(f"Error in classify_video_gemini: {e}")
        return "ERROR"


if __name__ == "__main__":
    main()
