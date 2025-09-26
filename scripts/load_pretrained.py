import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from models.revgg_r2net import create_revgg_r2net
from utils.metrics import get_custom_objects
from config import Config

config = Config()

def find_all_model_files(search_path="pretrained"):
    model_files = {}
    search_dirs = [search_path, "pretrained", ".", "models"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        print(f"Searching in: {search_dir}")
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"  Found: {file}")
                
                if file.endswith('.keras'):
                    model_files['keras'] = file_path
                elif file.endswith('.h5') and ('weights' in file.lower() or 'model' in file.lower()):
                    model_files['weights_or_model'] = file_path
                elif file.endswith('.h5'):
                    model_files['h5_file'] = file_path
                elif file == 'config.json':
                    model_files['config'] = file_path
                elif file == 'metadata.json':
                    model_files['metadata'] = file_path
    
    return model_files

def load_pretrained_model(pretrained_dir="pretrained"):
    model_files = find_all_model_files(pretrained_dir)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {pretrained_dir}")
    
    print("Available files:")
    for key, path in model_files.items():
        print(f"  {key}: {os.path.basename(path)}")
    
    custom_objects = get_custom_objects()
    
    load_attempts = [
        ('keras', lambda: tf.keras.models.load_model(model_files['keras'], custom_objects=custom_objects)),
        ('weights_or_model', lambda: load_h5_file(model_files['weights_or_model'], custom_objects)),
        ('h5_file', lambda: load_h5_file(model_files['h5_file'], custom_objects))
    ]
    
    for file_type, load_func in load_attempts:
        if file_type in model_files:
            try:
                print(f"Attempting to load {file_type}: {model_files[file_type]}")
                model = load_func()
                print(f"Successfully loaded model from {file_type}")
                return model
            except Exception as e:
                print(f"Failed to load {file_type}: {str(e)}")
                continue
    
    raise ValueError("Could not load model from any available files")

def load_h5_file(file_path, custom_objects):
    try:
        model = tf.keras.models.load_model(file_path, custom_objects=custom_objects)
        return model
    except:
        print("Failed as full model, trying as weights...")
        model = create_revgg_r2net()
        model.load_weights(file_path)
        return model

def preprocess_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            img_resized = img.resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            image_array = np.array(img_resized, dtype=np.float32)
        
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array]*3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        image_array = image_array / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}")
        return None

def get_image_files(directory):
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return []
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif', '.PNG', '.JPG', '.JPEG', '.TIFF', '.BMP', '.TIF')
    image_files = []
    
    all_files = os.listdir(directory)
    print(f"All files in {directory}: {all_files}")
    
    for file in all_files:
        if file.endswith(valid_extensions) and not file.startswith('.'):
            image_files.append(file)
            print(f"Valid image: {file}")
    
    if not image_files:
        print(f"No valid images found. Looking for: {valid_extensions}")
        
        subfolders = [f for f in all_files if os.path.isdir(os.path.join(directory, f))]
        if subfolders:
            print(f"Found subfolders: {subfolders}")
            for subfolder in subfolders:
                subfolder_path = os.path.join(directory, subfolder)
                sub_images = get_image_files(subfolder_path)
                if sub_images:
                    image_files.extend([os.path.join(subfolder, img) for img in sub_images])
    
    print(f"Found {len(image_files)} valid images")
    return image_files

def inference_with_pretrained(image_path, pretrained_dir="pretrained", threshold=0.5, output_dir=None):
    try:
        model = load_pretrained_model(pretrained_dir)
        print(f"Model loaded successfully. Input shape: {model.input_shape}")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
    
    if os.path.isfile(image_path):
        print(f"Processing single image: {image_path}")
        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            return None, None
        
        prediction = model.predict(preprocessed_image, verbose=0)
        binary_prediction = (prediction > threshold).astype(np.uint8)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            prob_mask = (prediction[0].squeeze() * 255).astype(np.uint8)
            binary_mask = (binary_prediction[0].squeeze() * 255).astype(np.uint8)
            
            Image.fromarray(prob_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_probability.png")
            )
            Image.fromarray(binary_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_binary.png")
            )
            print(f"Results saved to {output_dir}")
        
        return prediction[0], binary_prediction[0]
    
    elif os.path.isdir(image_path):
        print(f"Processing directory: {image_path}")
        image_files = get_image_files(image_path)
        
        if not image_files:
            return {
                'filenames': [],
                'predictions': np.array([]),
                'binary_predictions': np.array([])
            }
        
        predictions = []
        binary_predictions = []
        successful_files = []
        
        for image_file in image_files:
            full_path = os.path.join(image_path, image_file)
            print(f"Processing: {image_file}")
            
            preprocessed_image = preprocess_image(full_path)
            if preprocessed_image is not None:
                prediction = model.predict(preprocessed_image, verbose=0)
                binary_prediction = (prediction > threshold).astype(np.uint8)
                
                predictions.append(prediction[0])
                binary_predictions.append(binary_prediction[0])
                successful_files.append(image_file)
        
        results = {
            'filenames': successful_files,
            'predictions': np.array(predictions) if predictions else np.array([]),
            'binary_predictions': np.array(binary_predictions) if binary_predictions else np.array([])
        }
        
        if output_dir and successful_files:
            os.makedirs(output_dir, exist_ok=True)
            for filename, prediction, binary_pred in zip(successful_files, predictions, binary_predictions):
                base_name = os.path.splitext(filename)[0]
                
                prob_mask = (prediction.squeeze() * 255).astype(np.uint8)
                binary_mask = (binary_pred.squeeze() * 255).astype(np.uint8)
                
                Image.fromarray(prob_mask, mode='L').save(
                    os.path.join(output_dir, f"{base_name}_probability.png")
                )
                Image.fromarray(binary_mask, mode='L').save(
                    os.path.join(output_dir, f"{base_name}_binary.png")
                )
            print(f"Results saved to {output_dir}")
        
        print(f"Successfully processed {len(successful_files)} images")
        return results
    
    else:
        raise ValueError(f"Path not found: {image_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python load_pretrained.py <image_path> [pretrained_dir] [threshold] [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    pretrained_dir = sys.argv[2] if len(sys.argv) > 2 else "pretrained"
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "results"
    
    results = inference_with_pretrained(image_path, pretrained_dir, threshold, output_dir)
    
    if isinstance(results, dict):
        print(f"Processed {len(results['filenames'])} images")
        if len(results['filenames']) > 0:
            print(f"Sample prediction shape: {results['predictions'][0].shape}")
    else:
        print("Single image processed")
    
    print("Inference completed!")
