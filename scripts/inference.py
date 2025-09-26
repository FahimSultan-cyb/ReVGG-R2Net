import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from config import Config
from utils.metrics import get_custom_objects
from models.revgg_r2net import create_revgg_r2net

config = Config()

def find_model_files(search_path):
    model_files = {}
    search_dirs = []
    
    if os.path.isfile(search_path):
        search_dirs = [os.path.dirname(search_path)]
    elif os.path.isdir(search_path):
        search_dirs = [search_path]
    else:
        search_dirs = ["pretrained", ".", "models"]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.keras'):
                    model_files['keras'] = file_path
                elif file.endswith('.h5') and 'weights' in file.lower():
                    model_files['weights'] = file_path
                elif file.endswith('.h5'):
                    model_files['h5'] = file_path
    
    return model_files

def load_model_robust(model_path):
    model_files = find_model_files(model_path)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found. Searched in: {model_path}")
    
    print("Available model files:")
    for key, path in model_files.items():
        print(f"  {key}: {path}")
    
    if 'keras' in model_files:
        try:
            print(f"Loading .keras model: {model_files['keras']}")
            custom_objects = get_custom_objects()
            model = tf.keras.models.load_model(model_files['keras'], custom_objects=custom_objects)
            print("Model loaded successfully from .keras file")
            return model
        except Exception as e:
            print(f"Failed to load .keras file: {e}")
    
    if 'weights' in model_files:
        try:
            print(f"Loading weights: {model_files['weights']}")
            model = create_revgg_r2net()
            model.load_weights(model_files['weights'])
            print("Model loaded successfully from weights file")
            return model
        except Exception as e:
            print(f"Failed to load weights: {e}")
    
    if 'h5' in model_files:
        try:
            print(f"Loading .h5 model: {model_files['h5']}")
            custom_objects = get_custom_objects()
            model = tf.keras.models.load_model(model_files['h5'], custom_objects=custom_objects)
            print("Model loaded successfully from .h5 file")
            return model
        except Exception as e:
            print(f"Failed to load .h5 file: {e}")
            try:
                print("Trying to load as weights...")
                model = create_revgg_r2net()
                model.load_weights(model_files['h5'])
                print("Model loaded successfully as weights")
                return model
            except Exception as e2:
                print(f"Also failed as weights: {e2}")
    
    raise ValueError("Could not load model from any available files")

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model_robust(model_path)
    
    def _preprocess_image(self, image_path):
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
            print(f"Error preprocessing image {image_path}: {e}")
            return np.zeros((1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.INPUT_CHANNELS), dtype=np.float32)
    
    def predict_single(self, image_path, threshold=0.5):
        preprocessed_image = self._preprocess_image(image_path)
        prediction = self.model.predict(preprocessed_image, verbose=0)
        binary_prediction = (prediction > threshold).astype(np.uint8)
        return prediction[0], binary_prediction[0]
    
    def predict_batch(self, image_paths, threshold=0.5):
        predictions = []
        binary_predictions = []
        
        for image_path in image_paths:
            pred, binary_pred = self.predict_single(image_path, threshold)
            predictions.append(pred)
            binary_predictions.append(binary_pred)
        
        return np.array(predictions), np.array(binary_predictions)
    
    def predict_directory(self, image_dir, threshold=0.5):
        if not os.path.isdir(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif')
        image_files = []
        
        for file in os.listdir(image_dir):
            if not file.startswith('.') and file.lower().endswith(valid_extensions):
                image_files.append(file)
        
        if not image_files:
            print(f"No valid image files found in {image_dir}")
            print(f"Looking for files with extensions: {valid_extensions}")
            all_files = os.listdir(image_dir)
            print(f"Found files: {all_files}")
            return {
                'filenames': [],
                'predictions': np.array([]),
                'binary_predictions': np.array([])
            }
        
        print(f"Processing {len(image_files)} images...")
        image_paths = [os.path.join(image_dir, f) for f in image_files]
        
        predictions, binary_predictions = self.predict_batch(image_paths, threshold)
        
        return {
            'filenames': image_files,
            'predictions': predictions,
            'binary_predictions': binary_predictions
        }
    
    def save_predictions(self, predictions_dict, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (filename, prediction, binary_pred) in enumerate(zip(
            predictions_dict['filenames'],
            predictions_dict['predictions'],
            predictions_dict['binary_predictions']
        )):
            base_name = os.path.splitext(filename)[0]
            
            prob_mask = (prediction.squeeze() * 255).astype(np.uint8)
            binary_mask = (binary_pred.squeeze() * 255).astype(np.uint8)
            
            Image.fromarray(prob_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_probability.png")
            )
            Image.fromarray(binary_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_binary.png")
            )

def load_model_and_predict(model_path, image_path, threshold=0.5, output_dir=None):
    inference = ModelInference(model_path)
    
    if os.path.isfile(image_path):
        prediction, binary_prediction = inference.predict_single(image_path, threshold)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            prob_mask = (prediction.squeeze() * 255).astype(np.uint8)
            binary_mask = (binary_prediction.squeeze() * 255).astype(np.uint8)
            
            Image.fromarray(prob_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_probability.png")
            )
            Image.fromarray(binary_mask, mode='L').save(
                os.path.join(output_dir, f"{base_name}_binary.png")
            )
        
        return prediction, binary_prediction
    
    elif os.path.isdir(image_path):
        results = inference.predict_directory(image_path, threshold)
        
        if output_dir:
            inference.save_predictions(results, output_dir)
        
        return results
    
    else:
        raise ValueError("image_path must be a valid file or directory path")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path> [threshold] [output_dir]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    output_dir = sys.argv[4] if len(sys.argv) > 4 else None
    
    results = load_model_and_predict(model_path, image_path, threshold, output_dir)
    print("Inference completed successfully!")
