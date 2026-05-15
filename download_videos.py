import os
import shutil
import kagglehub

def download_and_setup_dataset():
    target_dir = "./data/depvidmood"
    
    print("======================================================")
    print("🎬 Downloading DepVidMood Video Dataset from Kaggle...")
    print("======================================================")
    
    # 1. Download using kagglehub (this handles the Kaggle API automatically)
    try:
        cached_path = kagglehub.dataset_download("ziya07/depvidmood-facial-expression-video-dataset")
        print(f"\n[+] Download complete! Cached at: {cached_path}")
    except Exception as e:
        print(f"\n[-] Failed to download. Ensure your Kaggle API key is configured properly. Error: {e}")
        return

    # 2. Prepare the local data directory
    print(f"\n[!] Moving files to {target_dir} ...")
    os.makedirs(target_dir, exist_ok=True)

    # 3. Copy files from the hidden cache to your project's data folder
    file_count = 0
    for item in os.listdir(cached_path):
        source_item = os.path.join(cached_path, item)
        destination_item = os.path.join(target_dir, item)
        
        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            file_count += len(os.listdir(source_item))
        else:
            shutil.copy2(source_item, destination_item)
            file_count += 1

    print("======================================================")
    print(f"✅ Success! Moved {file_count} files/folders to {target_dir}")
    print("You can now run: python video_evaluator.py")
    print("======================================================")

if __name__ == "__main__":
    download_and_setup_dataset()