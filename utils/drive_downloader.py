import os
import zipfile
import requests
from urllib.parse import urlparse, parse_qs
import gdown

def download_pretrained_models(download_dir="pretrained"):
    """
    Download pretrained ReVGG-R2Net models from Google Drive
    """
    os.makedirs(download_dir, exist_ok=True)
    
    drive_folder_url = "https://drive.google.com/file/d/1rtUQdYNGQKWEeAUsmLxBT1eYCdDXLyvB/view?usp=sharing"
    folder_id = "1rtUQdYNGQKWEeAUsmLxBT1eYCdDXLyvB"
    
    print(f"Downloading pretrained models to: {download_dir}")
    print("This may take a few minutes depending on your connection...")
    
    try:
        output_path = os.path.join(download_dir, "pretrained_models.zip")
        
        folder_url = f"https://drive.google.com/uc?id={folder_id}"
        gdown.download(folder_url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            print("Extracting downloaded files...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
            os.remove(output_path)
            print(f"✓ Pretrained models downloaded successfully to {download_dir}")
            
            downloaded_files = os.listdir(download_dir)
            print("Downloaded files:")
            for file in downloaded_files:
                print(f"  - {file}")
            
            return True
            
    except Exception as e:
        print(f"Download failed with gdown: {e}")
        print("Trying alternative method...")
        
        try:
            alternative_download(folder_id, download_dir)
            return True
        except Exception as e2:
            print(f"Alternative download also failed: {e2}")
            print("Manual download required.")
            print(f"Please visit: {drive_folder_url}")
            return False

def alternative_download(folder_id, download_dir):
    """Alternative download method"""
    import subprocess
    
    try:
        cmd = [
            "gdown", 
            "--folder", 
            f"https://drive.google.com/drive/folders/{folder_id}",
            "-O", download_dir
        ]
        subprocess.run(cmd, check=True)
        print("✓ Downloaded using alternative method")
    except subprocess.CalledProcessError:
        raise Exception("Alternative download method failed")

def download_single_file(file_id, output_path, file_name=None):
    """Download a single file from Google Drive"""
    if file_name is None:
        file_name = os.path.basename(output_path)
    
    print(f"Downloading {file_name}...")
    
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        
        if os.path.exists(output_path):
            print(f"✓ {file_name} downloaded successfully")
            return True
        else:
            print(f"✗ Failed to download {file_name}")
            return False
            
    except Exception as e:
        print(f"Download error: {e}")
        return False

def setup_pretrained_models():
    """Setup pretrained models with automatic download"""
    pretrained_dir = "pretrained"
    
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir)
    
    existing_files = [f for f in os.listdir(pretrained_dir) if f.endswith(('.keras', '.h5', '.weights'))]
    
    if existing_files:
        print("Found existing pretrained models:")
        for file in existing_files:
            print(f"  - {file}")
        
        download_choice = input("Download additional models? (y/n): ").lower().strip()
        if download_choice != 'y':
            return existing_files
    
    success = download_pretrained_models(pretrained_dir)
    
    if success:
        model_files = [f for f in os.listdir(pretrained_dir) if f.endswith(('.keras', '.h5', '.weights'))]
        return model_files
    else:
        return []

if __name__ == "__main__":
    print("ReVGG-R2Net Pretrained Model Downloader")
    print("="*50)
    
    models = setup_pretrained_models()
    
    if models:
        print(f"\n✓ Available pretrained models: {len(models)}")
        for model in models:
            print(f"  - {model}")
    else:
        print("\n⚠ No pretrained models available")
        print("Please check your internet connection and try again")
