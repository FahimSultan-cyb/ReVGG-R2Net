#!/usr/bin/env python3

print("ReVGG-R2Net Quick Start V5")
print("="*50)

import os
import sys

def setup_environment():
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        print("Environment configured")
    except:
        print("TensorFlow not installed")

def quick_training_example():
    print("\n1. TRAINING EXAMPLE")
    print("-" * 30)
    print("from scripts.train import train_model")
    print('model, history = train_model("/path/to/your/dataset")')
    print("# Returns 2 values - model and history")

def quick_inference_example():
    print("\n2. INFERENCE EXAMPLE (RECOMMENDED)")
    print("-" * 30)
    print("from scripts.load_pretrained import inference_with_pretrained")
    print("")
    print("results = inference_with_pretrained(")
    print('    image_path="/content/drive/MyDrive/test",')
    print('    pretrained_dir="pretrained",')
    print('    threshold=0.5,')
    print('    output_dir="results"')
    print(")")
    print("# Auto-finds model files, processes all images")

def quick_evaluation_example():
    print("\n3. EVALUATION EXAMPLE")
    print("-" * 30)
    print("from scripts.evaluate import evaluate_model")
    print("")
    print("results = evaluate_model(")
    print('    model_path="pretrained",  # Auto-finds files')
    print('    test_data_path="/path/to/test/data"')
    print(")")

def download_example():
    print("\n4. DOWNLOAD PRETRAINED MODELS")
    print("-" * 30)
    print("!python download_pretrained.py")
    print("# Downloads models to pretrained/ directory")

def troubleshooting():
    print("\n5. TROUBLESHOOTING")
    print("-" * 30)
    print("Common Issues:")
    print("- Empty results: Check image file extensions")
    print("- Model not found: Run download_pretrained.py first")
    print("- Memory error: Use train.py instead of train.py")
    print("- Path error: Use full absolute paths")

def main():
    setup_environment()
    quick_training_example()
    quick_inference_example() 
    quick_evaluation_example()
    download_example()
    troubleshooting()
    
    print("\n" + "="*50)
    print("COPY-PASTE READY CODE:")
    print("="*50)
    print("# Step 1: Setup")
    print("!git clone https://github.com/FahimSultan-cyb/ReVGG-R2Net.git")
    print("import sys")
    print("import os, sys")
print(f"root_path = os.path.join(os.getcwd(), 'ReVGG-R2Net')")
    print("os.chdir(root_path)")
    print("")
    print("# Step 2: Download models")
    print("!python download_pretrained.py")
    print("")
    print("# Step 3: Run inference")
    print("from scripts.load_pretrained import inference_with_pretrained")
    print("results = inference_with_pretrained('/your/image/path', 'pretrained')")
    print("")
    print("# Step 4: Check results")
    print("print(f'Processed {len(results[\"filenames\"])} images')")

if __name__ == "__main__":
    main()
