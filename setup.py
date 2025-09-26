#!/usr/bin/env python3

import os
import sys
import subprocess
import tensorflow as tf

def configure_environment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")
    
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

def install_requirements():
    requirements = [
        'tensorflow>=2.10.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'pillow>=8.3.0',
        'matplotlib>=3.5.0',
        'scikit-image>=0.19.0',
        'gdown>=4.6.0'
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Installed: {req}")
        except subprocess.CalledProcessError:
            print(f"Failed to install: {req}")

def test_installation():
    try:
        configure_environment()
        
        from models.revgg_r2net import create_revgg_r2net
        from utils.metrics import dice_coef
        from utils.data_loader import DataLoader
        
        print("Testing model creation...")
        model = create_revgg_r2net()
        print(f"Model created: {model.count_params():,} parameters")
        
        del model
        tf.keras.backend.clear_session()
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    print("ReVGG-R2Net Setup V5")
    print("="*40)
    
    print("Installing requirements...")
    install_requirements()
    
    print("Testing installation...")
    if test_installation():
        print("\nSetup completed successfully!")
        print("Usage:")
        print("  Training: python scripts/train.py /path/to/data")
        print("  Inference: python scripts/load_pretrained.py /path/to/images")
        print("  Evaluate: python scripts/evaluate.py pretrained /path/to/test")
    else:
        print("\nSetup failed. Check dependencies.")

if __name__ == "__main__":
    main()
