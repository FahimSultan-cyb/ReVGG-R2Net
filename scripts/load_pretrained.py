import os
import json
import tensorflow as tf
from models.revgg_r2net import create_revgg_r2net
from utils.metrics import get_custom_objects
import numpy as np
def find_model_files(pretrained_dir="pretrained"):
    model_files = {}
    
    for root, dirs, files in os.walk(pretrained_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.keras'):
                model_files['keras'] = file_path
            elif file.endswith('.h5'):
                model_files['weights'] = file_path
            elif file == 'config.json':
                model_files['config'] = file_path
    
    return model_files

def load_pretrained_model(pretrained_dir="pretrained"):
    model_files = find_model_files(pretrained_dir)
    
    print("Found files:")
    for key, path in model_files.items():
        print(f"  {key}: {path}")
    
    if 'keras' in model_files and os.path.exists(model_files['keras']):
        print(f"Loading .keras model: {model_files['keras']}")
        try:
            custom_objects = get_custom_objects()
            model = tf.keras.models.load_model(model_files['keras'], custom_objects=custom_objects)
            print("Model loaded successfully from .keras file")
            return model
        except Exception as e:
            print(f"Failed to load .keras file: {e}")
    
    if 'weights' in model_files and os.path.exists(model_files['weights']):
        print(f"Loading model weights: {model_files['weights']}")
        try:
            model = create_revgg_r2net()
            model.load_weights(model_files['weights'])
            print("✓ Model loaded successfully from weights file")
            return model
        except Exception as e:
            print(f"Failed to load weights: {e}")
    
    raise ValueError("No valid model files found in pretrained directory")

def get_model_path(pretrained_dir="pretrained"):
    model_files = find_model_files(pretrained_dir)
    
    if 'keras' in model_files:
        return model_files['keras']
    elif 'weights' in model_files:
        return model_files['weights']
    else:
        available_files = []
        for root, dirs, files in os.walk(pretrained_dir):
            for file in files:
                available_files.append(os.path.join(root, file))
        
        print("Available files in pretrained directory:")
        for file in available_files:
            print(f"  - {file}")
        
        raise FileNotFoundError(f"No model files found in {pretrained_dir}")

def inference_with_pretrained(image_path, pretrained_dir="pretrained", threshold=0.5, output_dir=None):
    try:
        model = load_pretrained_model(pretrained_dir)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None
    
    from scripts.inference import ModelInference
    
    class PretrainedInference:
        def __init__(self, model):
            self.model = model
        
        def _preprocess_image(self, image_path):
            import numpy as np
            from PIL import Image, ImageOps
            from config import Config
            config = Config()
            
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
                return None
        
        def predict_single(self, image_path, threshold=0.5):
            preprocessed_image = self._preprocess_image(image_path)
            if preprocessed_image is None:
                return None, None
            
            prediction = self.model.predict(preprocessed_image, verbose=0)
            binary_prediction = (prediction > threshold).astype(np.uint8)
            return prediction[0], binary_prediction[0]
        
        def predict_directory(self, image_dir, threshold=0.5):
            image_files = [f for f in os.listdir(image_dir) 
                          if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            
            predictions = []
            binary_predictions = []
            
            for image_file in image_files:
                image_path = os.path.join(image_dir, image_file)
                pred, binary_pred = self.predict_single(image_path, threshold)
                if pred is not None:
                    predictions.append(pred)
                    binary_predictions.append(binary_pred)
            
            return {
                'filenames': image_files,
                'predictions': np.array(predictions),
                'binary_predictions': np.array(binary_predictions)
            }
        
        def save_predictions(self, predictions_dict, output_dir):
            import numpy as np
            from PIL import Image
            
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
    
    inference = PretrainedInference(model)
    
    if os.path.isfile(image_path):
        prediction, binary_prediction = inference.predict_single(image_path, threshold)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            prob_mask = (prediction.squeeze() * 255).astype(np.uint8)
            binary_mask = (binary_prediction.squeeze() * 255).astype(np.uint8)
            
            from PIL import Image
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
    if len(sys.argv) < 2:
        print("Usage: python load_pretrained.py <image_path> [threshold] [output_dir]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    results = inference_with_pretrained(image_path, "pretrained", threshold, output_dir)
    print("Inference completed!")
