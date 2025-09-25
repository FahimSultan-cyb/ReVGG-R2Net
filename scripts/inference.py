import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from config import Config
from utils.metrics import get_custom_objects

config = Config()

class ModelInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        print(f"Loading model from: {self.model_path}")
        custom_objects = get_custom_objects()
        model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)
        return model
    
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
        image_files = [f for f in os.listdir(image_dir) 
                      if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
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
