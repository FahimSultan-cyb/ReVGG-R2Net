

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    requirements = [
        'tensorflow>=2.10.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'pillow>=8.3.0',
        'matplotlib>=3.5.0',
        'scikit-image>=0.19.0'
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError:
            print(f"Failed to install {req}")

def configure_environment():
    """Configure environment for optimal performance"""
    import tensorflow as tf
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found, using CPU")
    
    # Set environment variables for memory optimization
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    
    print("✓ Environment configured for memory optimization")

def test_installation():
    """Test if everything is installed correctly"""
    try:
        import tensorflow as tf
        import numpy as np
        from PIL import Image
        import sklearn
        import matplotlib.pyplot as plt
        
        print("✓ All packages imported successfully")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"NumPy version: {np.__version__}")
        
        # Test model creation
        sys.path.append(os.path.dirname(__file__))
        from models.revgg_r2net import create_revgg_r2net
        
        model = create_revgg_r2net()
        print(f"✓ Model created successfully with {model.count_params():,} parameters")
        
        del model
        tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("ReVGG-R2Net Setup for Colab/Jupyter")
    print("=" * 60)
    
    # Install requirements
    print("\n1. Installing requirements...")
    install_requirements()
    
    # Configure environment
    print("\n2. Configuring environment...")
    configure_environment()
    
    # Test installation
    print("\n3. Testing installation...")
    if test_installation():
        print("\n" + "=" * 60)
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now use:")
        print("- scripts/train_colab.py for training")
        print("- notebooks/simple_training.ipynb for Jupyter/Colab")
        print("- Use batch_size=1 for memory-constrained environments")
    else:
        print("\n" + "=" * 60)
        print("✗ SETUP FAILED!")
        print("=" * 60)
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
