#!/usr/bin/env python3

import os
import sys
import subprocess

def install_gdown():

    try:
        import gdown

        return True
    except ImportError:

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

            return True
        except Exception as e:

            return False

def download_from_google_drive():

    
    if not install_gdown():
        return False
    
    import gdown
    
    folder_id = "1338RpeFgrQAA20N9cvMKzQt58dPDriC_"
    download_dir = "pretrained"
    
    os.makedirs(download_dir, exist_ok=True)
    
    print("="*60)
    print("Downloading ReVGG-R2Net Pretrained Models")
    print("="*60)
    print("This may take several minutes...")
    print(f"Source: https://drive.google.com/drive/folders/{folder_id}")
    print(f"Destination: {download_dir}/")

    
    try:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(folder_url, output=download_dir, quiet=False, use_cookies=False)
        
        downloaded_files = []
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if file.endswith(('.keras', '.h5', '.weights')):
                    downloaded_files.append(os.path.join(root, file))
        
        if downloaded_files:
            print("\n" + "="*60)
            print("DOWNLOAD COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Downloaded model files:")
            for file in downloaded_files:
                file_size = os.path.getsize(file) / (1024*1024)  # MB
                print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
            
            print(f"\nModels are ready to use in the '{download_dir}' directory!")
            return True
        else:
            print("No model files found in downloaded content")
            return False
            
    except Exception as e:
        print("This may take several minutes...\n\n")
        
        print("You can Download it manually and place in 'pretrained/' directory\n\n")

        print("Visit: https://drive.google.com/file/d/1rtUQdYNGQKWEeAUsmLxBT1eYCdDXLyvB/view?usp=drive_link")

        return False

def main():
    print("ReVGG-R2Net Pretrained Model Downloader")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":

        print("Models will be saved in the 'pretrained/' directory")
        return
    
    success = download_from_google_drive()
    
    if success:
        print("\n Ready to use ReVGG-R2Net!")

    else:
        print("This may take several minutes...")

if __name__ == "__main__":
    main()
